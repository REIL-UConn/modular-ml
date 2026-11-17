"""Wrap functional JAX models and stateless Flax Linen modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from modularml.core.models.jax_base_model import JaxBaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.environment.optional_imports import ensure_jax


class JaxModelWrapper(JaxBaseModel):
    """
    Wrap ``model(params, x)`` or a module exposing ``init`` and ``apply``.

    Functions require explicit ``params`` or ``init_fn(key, x)``. Flax Linen
    modules initialize lazily at build time; their params collection is optimized
    and other immutable variable collections are retained for inference.
    Stateful/mutable collections and stochastic apply calls require a custom
    JaxBaseModel implementation.
    """

    def __init__(
        self,
        model,
        *,
        params: Any = None,
        init_fn=None,
        seed: int = 0,
        **kwargs,
    ):
        if not callable(model) and not callable(getattr(model, "apply", None)):
            raise TypeError("model must be callable or expose an apply method.")
        if init_fn is not None and not callable(init_fn):
            raise TypeError("init_fn must be callable.")
        super().__init__(model=model, init_fn=init_fn, seed=seed, **kwargs)
        self.model = model
        self.init_fn = init_fn
        self.seed = seed
        self.params = params
        self._initial_params = params
        self._variables = {}
        self._input_shape = None
        self._output_shape = None

    @property
    def input_shape(self):
        """Return per-sample input dimensions."""
        return self._input_shape

    @property
    def output_shape(self):
        """Return per-sample output dimensions."""
        return self._output_shape

    def build(self, input_shape=None, output_shape=None, *, force=False):
        """Initialize parameters and validate output shape with a dummy batch."""
        if self.is_built and not force:
            return
        if input_shape is None:
            raise ValueError("input_shape is required to build a JAX model.")
        jax = ensure_jax()
        shape = ensure_tuple_shape(input_shape, min_len=1, allow_null_shape=False)
        dummy = jax.numpy.zeros((1, *shape), dtype=jax.numpy.float32)
        initializer = self.init_fn or getattr(self.model, "init", None)
        if force:
            self.params = None if initializer is not None else self._initial_params
        if self.params is None:
            if initializer is None:
                raise ValueError(
                    "Provide params or init_fn for a functional JAX model.",
                )
            initialized = initializer(jax.random.PRNGKey(self.seed), dummy)
            if callable(getattr(self.model, "apply", None)):
                self.params = initialized.get("params", {})
                self._variables = {
                    k: v for k, v in initialized.items() if k != "params"
                }
            else:
                self.params = initialized
        self.params = jax.tree_util.tree_map(jax.numpy.asarray, self.params)
        inferred = tuple(self.forward(dummy).shape[1:])
        if output_shape is not None and tuple(output_shape) != inferred:
            msg = f"Expected output shape {tuple(output_shape)}, got {inferred}."
            raise ValueError(msg)
        self._input_shape = shape
        self._output_shape = inferred
        self._initial_params = self.get_weights()["params"]
        self._built = True

    def forward(self, x, **kwargs):
        """Apply the model with the current parameter tree."""
        if self.params is None:
            raise RuntimeError("Build the JAX model before calling forward.")
        x = ensure_jax().numpy.asarray(x)
        if callable(getattr(self.model, "apply", None)):
            return self.model.apply(
                {**self._variables, "params": self.params},
                x,
                **kwargs,
            )
        return self.model(self.params, x, **kwargs)

    def get_init_args(self):
        """Include initial parameters to reconstruct eagerly initialized models."""
        return {**super().get_init_args(), "params": self._initial_params}

    def get_config(self):
        """Encode initial arrays as JSON data rather than code symbols."""
        cfg = super().get_config()
        cfg["init_args"]["params"] = ensure_jax().tree_util.tree_map(
            lambda x: {
                "__jax_array__": np.asarray(x).tolist(),
                "dtype": str(np.asarray(x).dtype),
            },
            self._initial_params,
        )
        cfg["variables"] = ensure_jax().tree_util.tree_map(
            lambda x: {
                "__jax_array__": np.asarray(x).tolist(),
                "dtype": str(np.asarray(x).dtype),
            },
            self._variables,
        )
        if callable(getattr(self.model, "apply", None)):
            from dataclasses import fields

            cfg["init_args"].pop("model")
            module_cls = type(self.model)
            cfg["module_class"] = (
                f"{module_cls.__module__}.{module_cls.__qualname__}"
                if module_cls.__module__.startswith("flax.")
                else module_cls
            )
            cfg["module_kwargs"] = {
                field.name: getattr(self.model, field.name)
                for field in fields(self.model)
                if field.init
                and field.name not in ("parent", "name")
                and getattr(self.model, field.name) is not field.default
            }
        return cfg

    @classmethod
    def from_config(cls, config):
        """Restore array metadata and reconstruct the wrapped function/module."""
        jax = ensure_jax()

        def decode(tree):
            return jax.tree_util.tree_map(
                lambda x: jax.numpy.asarray(x["__jax_array__"], dtype=x["dtype"]),
                tree,
                is_leaf=lambda x: isinstance(x, dict) and "__jax_array__" in x,
            )

        args = dict(config["init_args"])
        args["params"] = decode(args.get("params"))
        if "module_class" in config:
            module_cls = config["module_class"]
            if isinstance(module_cls, str):
                from modularml.utils.environment.environment import import_from_path

                module_cls = import_from_path(module_cls)
            args["model"] = module_cls(**config["module_kwargs"])
        model = cls(**args)
        model._variables = decode(config.get("variables", {}))
        if config.get("is_built"):
            model.build(config["input_shape"], config["output_shape"])
        return model

    def get_weights(self):
        """Include immutable Flax variable collections in saved state."""
        weights = super().get_weights()
        jax = ensure_jax()
        weights["variables"] = jax.tree_util.tree_map(
            lambda x: np.array(x, copy=True),
            self._variables,
        )
        return weights

    def set_weights(self, weights):
        """Restore parameters and immutable variable collections."""
        super().set_weights(weights)
        self._variables = ensure_jax().tree_util.tree_map(
            ensure_jax().numpy.asarray,
            weights.get("variables", self._variables),
        )
