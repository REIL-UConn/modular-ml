"""JAX backend integration with models, graph execution, and checkpoints."""

import numpy as np
import pytest

from modularml import (
    AppliedLoss,
    FeatureSet,
    JaxModelWrapper,
    Loss,
    ModelGraph,
    ModelNode,
    Optimizer,
)
from modularml.core.data.batch import Batch
from modularml.core.data.execution_context import ExecutionContext
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.utils.data.conversion import (
    align_ranks,
    convert_to_format,
    to_numpy,
    to_python,
)
from modularml.utils.data.data_format import DataFormat
from modularml.utils.nn.backend import (
    Backend,
    backend_requires_optimizer,
    infer_backend,
)

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
jnp = jax.numpy

pytestmark = pytest.mark.unit


def linear(params, x):
    return x @ params["w"] + params["b"]


def initialize(key, x):
    return {"w": jax.random.normal(key, (x.shape[-1], 1)), "b": jnp.zeros((1,))}


def make_model(value=0.0):
    return JaxModelWrapper(
        linear,
        params={"w": jnp.full((1, 1), value), "b": jnp.zeros(1)},
    )


def make_context(node, x=None, y=None):
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32) if x is None else x
    y = 2 * x if y is None else y
    sd = SampleData(features=x, targets=y)
    batch = Batch(
        batch_size=len(x),
        role_data=RoleData({"default": sd}),
        shapes=sd.shapes,
        role_weights={"default": np.ones(len(x), dtype=np.float32)},
        role_masks={"default": np.ones(len(x), dtype=np.int8)},
    )
    ctx = ExecutionContext("train", 0, 0)
    ctx.set_input(node_id=node.node_id, upstream=node.upstream_ref, batch=batch)
    return ctx


def setup_graph(*, global_optimizer=False, two_nodes=False):
    fs = FeatureSet.from_dict(
        label="data",
        data={"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]},
        feature_keys=["x"],
        target_keys=["y"],
    )
    opt = Optimizer("adam", backend=Backend.JAX, opt_kwargs={"learning_rate": 0.05})
    head = ModelNode(
        "head",
        make_model(0.5),
        fs,
        optimizer=None if global_optimizer else opt,
    )
    nodes = [head]
    if two_nodes:
        nodes.append(ModelNode("tail", make_model(0.5), head))
    graph = ModelGraph(nodes, optimizer=opt if global_optimizer else None)
    graph.build()
    loss = AppliedLoss(
        Loss("mse", backend=Backend.JAX),
        on=nodes[-1],
        inputs=["outputs", "targets"],
    )
    return graph, head, nodes[-1], loss


def test_wrapper_build_gradients_and_reset():
    model = JaxModelWrapper(linear, init_fn=initialize, seed=42)
    model.build((2,), (1,))
    original = model.get_weights()
    grads = jax.grad(lambda p: jnp.sum(linear(p, jnp.ones((3, 2)))))(model.params)
    np.testing.assert_allclose(grads["w"], 3)
    model.set_weights({"params": {"w": np.zeros((2, 1)), "b": np.zeros(1)}})
    model.reset_weights()
    np.testing.assert_array_equal(model.params["w"], original["params"]["w"])
    clone = JaxModelWrapper.from_config(model.get_config())
    clone.set_state(model.get_state())
    np.testing.assert_allclose(clone(jnp.ones((3, 2))), model(jnp.ones((3, 2))))


def test_data_conversion_preserves_autodiff():
    x = jnp.ones((3, 1))
    assert infer_backend(x) == Backend.JAX
    assert backend_requires_optimizer(Backend.JAX)
    assert isinstance(convert_to_format(np.ones((3, 1)), DataFormat.JAX), jax.Array)
    np.testing.assert_array_equal(to_numpy(x), np.ones((3, 1)))
    assert to_python(jnp.array(2.0)) == 2.0
    grad = jax.grad(lambda v: jnp.sum(convert_to_format(v, DataFormat.JAX)))(x)
    np.testing.assert_array_equal(grad, x)
    a, b = align_ranks(jnp.ones(3), x)
    assert a.shape == b.shape


@pytest.mark.parametrize(
    ("global_optimizer", "two_nodes"),
    [(False, False), (True, False), (True, True)],
)
def test_training_reduces_loss(global_optimizer, two_nodes):
    graph, head, tail, loss = setup_graph(
        global_optimizer=global_optimizer,
        two_nodes=two_nodes,
    )
    values = []
    for _ in range(15):
        ctx = make_context(head)
        graph.train_step(ctx, [loss])
        values.append(float(ctx.losses.trainable))
        assert not isinstance(ctx.outputs[tail.node_id].features, jax.core.Tracer)
    assert values[-1] < values[0]


def test_global_freeze_keeps_weights_and_allows_upstream_gradients():
    graph, head, tail, loss = setup_graph(global_optimizer=True, two_nodes=True)
    tail.freeze()
    frozen = tail.model.get_weights()
    initial = head.model.get_weights()
    graph.train_step(make_context(head), [loss])
    np.testing.assert_array_equal(tail.model.params["w"], frozen["params"]["w"])
    assert not np.array_equal(head.model.params["w"], initial["params"]["w"])


def test_optimizer_checkpoint_continues_identically():
    params = {"w": jnp.ones((2, 1))}
    grads = {"w": jnp.full((2, 1), 0.3)}
    opt = Optimizer("adam", backend=Backend.JAX, opt_kwargs={"learning_rate": 0.1})
    opt.build(parameters=params)
    params = opt.step(grads=grads, variables=params)
    clone = Optimizer.from_config(opt.get_config())
    clone.set_state(opt.get_state())
    clone.build(parameters=params)
    np.testing.assert_array_equal(
        opt.step(grads=grads, variables=params)["w"],
        clone.step(grads=grads, variables=params)["w"],
    )


def test_serialized_model_roundtrip(tmp_path):
    model = make_model(0.5)
    model.build((1,))
    path = model.save(tmp_path / "jax_model")
    clone = JaxModelWrapper.load(path, allow_packaged_code=True)
    np.testing.assert_array_equal(model(jnp.ones((3, 1))), clone(jnp.ones((3, 1))))


def test_validation_errors():
    with pytest.raises(TypeError):
        JaxModelWrapper(42)
    with pytest.raises(ValueError, match="params or init_fn"):
        JaxModelWrapper(linear).build((1,))
    with pytest.raises(ValueError, match="output shape"):
        make_model().build((1,), (2,))


def test_flax_module_auto_wrap_and_serialization(tmp_path):
    nn = pytest.importorskip("flax.linen")
    from modularml.core.models import wrap_model

    model = wrap_model(nn.Dense(1))
    assert isinstance(model, JaxModelWrapper)
    model.build((2,), (1,))
    clone = JaxModelWrapper.from_config(model.get_config())
    clone.set_state(model.get_state())
    np.testing.assert_array_equal(model(jnp.ones((2, 2))), clone(jnp.ones((2, 2))))
    path = model.save(tmp_path / "flax_model")
    clone = JaxModelWrapper.load(path, allow_packaged_code=True)
    np.testing.assert_array_equal(model(jnp.ones((2, 2))), clone(jnp.ones((2, 2))))


def test_eval_does_not_change_parameters():
    graph, head, _tail, loss = setup_graph()
    head.freeze()
    weights = head.model.get_weights()
    ctx = make_context(head)
    graph.eval_step(ctx, [loss])
    np.testing.assert_array_equal(weights["params"]["w"], head.model.params["w"])
    assert float(ctx.losses.auxiliary) > 0


def test_failure_restores_parameters_and_context():
    from modularml.utils.nn.jax_training import train_jax

    _graph, head, _tail, loss = setup_graph()
    ctx = make_context(head)
    params = head.model.params

    def failing_forward():
        ctx.outputs["temporary"] = head.model.params
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        train_jax(
            {head.node_id: head},
            head._optimizer,
            ctx,
            [loss],
            failing_forward,
            node_optimizer=True,
        )
    assert head.model.params is params
    assert ctx.outputs == {}


def test_cpu_accelerator_and_jax_shapes():
    from modularml import Accelerator
    from modularml.utils.data.shape_utils import get_shape

    graph, head, _tail, loss = setup_graph()
    ctx = make_context(head)
    graph.train_step(ctx, [loss], accelerator=Accelerator("cpu"))
    assert ctx.outputs[head.node_id].features.device.platform == "cpu"
    assert get_shape(ctx.outputs[head.node_id].features) == (3, 1)


def test_concat_preserves_gradient_and_padding():
    from modularml import ConcatNode

    _graph, head, tail, _loss = setup_graph(global_optimizer=True, two_nodes=True)
    node = ConcatNode(
        "merge",
        upstream_refs=[head, tail],
        concat_axis=0,
        pad_inputs=True,
    )
    x = jnp.ones((3, 2))
    result = jax.grad(
        lambda v: node.apply_merge([v, v[:, :1]], fmt=DataFormat.JAX).sum(),
    )(x)
    np.testing.assert_array_equal(result, jnp.array([[2.0, 1.0]] * 3))


def test_optional_import_errors(monkeypatch):
    from modularml.utils.environment import optional_imports

    monkeypatch.setattr(optional_imports, "check_jax", lambda: None)
    with pytest.raises(ImportError, match=r"modularml\[jax\]"):
        optional_imports.ensure_jax()
