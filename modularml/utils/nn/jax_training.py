"""Adapt ModularML execution containers to JAX's functional differentiation."""

from modularml.core.training.loss_record import LossCollection, LossRecord
from modularml.utils.environment.optional_imports import ensure_jax


def train_jax(nodes, optimizer, ctx, losses, forward, *, node_optimizer=False):
    """
    Differentiate parameters while restoring all temporary tracer bindings.

    Python graph traversal runs eagerly under value_and_grad. Only array values
    are differentiated; execution containers are restored before leaving tracing.
    """
    jax = ensure_jax()
    params = {nid: node.model.params for nid, node in nodes.items()}
    original_outputs = dict(ctx.outputs)

    def objective(parameters):
        try:
            for nid, node in nodes.items():
                node.model.params = parameters[nid]
            for nid, batch in forward().items():
                ctx.set_output(node_id=nid, batch=batch)
            values = tuple(loss.compute(ctx=ctx) for loss in losses)
            return sum(values, jax.numpy.array(0.0)), values
        finally:
            for nid, node in nodes.items():
                node.model.params = params[nid]
            ctx.outputs.clear()
            ctx.outputs.update(original_outputs)

    (_, values), grads = jax.value_and_grad(objective, has_aux=True)(params)
    # Materialize ordinary arrays before updating weights, keeping recorded outputs
    # and losses from the same forward pass.
    for nid, batch in forward().items():
        ctx.set_output(node_id=nid, batch=batch)
    if node_optimizer:
        nid = next(iter(nodes))
        updated = {nid: optimizer.step(grads=grads[nid], variables=params[nid])}
    else:
        updated = optimizer.step(grads=grads, variables=params)
    for nid, node in nodes.items():
        node.model.params = updated[nid]
    ctx.add_losses(
        LossCollection(
            records=[
                LossRecord(label=loss.label, node_id=loss.node_id, trainable=value)
                for loss, value in zip(losses, values, strict=True)
            ],
        ),
    )
