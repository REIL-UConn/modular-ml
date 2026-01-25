from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from modularml.core.io.handlers.handler import BaseHandler
from modularml.core.topology.model_graph import ModelGraph
from modularml.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from modularml.core.io.serializer import LoadContext

logger = get_logger(level=logging.INFO)


class ModelGraphHandler(BaseHandler[ModelGraph]):
    """
    BaseHandler for ModelGraph objects.

    Encodes:
        - graph structure (nodes + optimizer config)
        - runtime state (node state + optimizer state)
    """

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # Decoding
    # ================================================
    def decode(
        self,
        cls: type[ModelGraph],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> ModelGraph:
        """
        Decodes ModelGraph from a saved artifact.

        Description:
            Instantiates a ModelGraph (instantiates from config and sets state).

        Args:
            cls (type[ModelGraph]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            ModelGraph: The re-instantiated object.

        """
        from modularml.context.experiment_context import ExperimentContext
        from modularml.core.topology.graph_node import GraphNode
        from modularml.core.training.optimizer import Optimizer

        exp_ctx = ExperimentContext.get_active()

        # Decode config (do not register constructed nodes)
        json_cfg = self.decode_config(
            load_dir=load_dir,
            ctx=ctx,
            config_rel_path=self.config_rel_path,
        )
        cfg = self._restore_json_cfg(data=json_cfg, ctx=ctx, register=False)

        # ------------------------------------------------
        # Collision checking for each node
        # - If any collisions occur, overwrite_existing must be True
        # - Since ModelGraph requires connections between nodes, we
        #   cannot easily assign new node IDs to loaded nodes.
        # ------------------------------------------------
        decoded_nodes: list[GraphNode] = []
        for node_cfg in cfg["nodes"]:
            node = GraphNode.from_config(node_cfg, register=False)
            cls_name = type(node).__qualname__
            node_id = node_cfg["node_id"]

            # If matching node ID exists
            if exp_ctx.has_node(node_id=node_id):
                existing = exp_ctx.get_node(node_id=node_id)

                # If identical node, use existing
                if existing == node:
                    msg = (
                        f"Loaded {cls_name} is identical to '{existing.label}' in "
                        f"the existing ExperimentContext. Node '{existing}' will be "
                        "reused."
                    )
                    logger.info(
                        msg=msg,
                        extra={
                            "title_desc": "Node ID Collision",
                            "omit_origin": True,
                        },
                    )

                    # Append existing node and continute (already registered)
                    decoded_nodes.append(existing)
                    continue

                # If not, overwrite_collison must be True
                if ctx.overwrite_collision:
                    msg = (
                        f"The loaded {cls_name} has an overlapping ID with existing "
                        f"{cls_name} '{existing.label}'. '{existing.label}' will be "
                        "overwritten in the active ExperimentContext."
                    )
                    logger.info(
                        msg=msg,
                        extra={
                            "title_desc": "Node ID Collision",
                            "omit_origin": True,
                        },
                    )
                    _ = exp_ctx.remove_node(
                        node_id=existing.node_id,
                        error_if_missing=True,
                    )

                    # Append new node and register
                    decoded_nodes.append(node)
                    exp_ctx.register_experiment_node(
                        node=node,
                        check_label_collision=False,
                    )
                    continue

                msg = (
                    f"The loaded {cls_name} has the same node ID as an "
                    f"existing node ('{existing.label}') in the active "
                    "ExperimentContext. Set `overwrite=True` to replace it "
                    "or create a new context to load the ModelGraph into."
                )
                raise RuntimeError(msg)

            # Otherwise, append new node and register
            decoded_nodes.append(node)
            exp_ctx.register_experiment_node(
                node=node,
                check_label_collision=False,
            )

        # Decode optimizer (if defined)
        optimizer = None
        if cfg.get("optimizer") is not None:
            optimizer = Optimizer.from_config(cfg["optimizer"])

        # Instantiate ModelGraph & restore state
        mg = cls(
            nodes=decoded_nodes,
            optimizer=optimizer,
            label=cfg.get("label", "model-graph"),
            ctx=exp_ctx,
            register=False,
        )
        state = self.decode_state(load_dir=load_dir, ctx=ctx)
        mg.set_state(state=state)

        # Collision checking of model graph
        existing = exp_ctx.model_graph
        if existing is not None:
            # Case 1: loaded ModelGraph is identical to existing, early return
            if existing == mg:
                msg = (
                    f"Loaded ModelGraph is identical to '{existing.label}' in "
                    f"the existing ExperimentContext. Returning '{existing.label}'."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "ModelGraph Collision",
                        "omit_origin": True,
                    },
                )
                return existing

            # Case 2: loaded graph differs -> overwrite
            if ctx.overwrite_collision:
                msg = (
                    f"The existing ModelGraph '{existing.label}' will be "
                    "overwritten with the loaded ModelGraph."
                )
                logger.info(
                    msg=msg,
                    extra={
                        "title_desc": "ModelGraph Collision",
                        "omit_origin": True,
                    },
                )
                exp_ctx.remove_model_graph()
                exp_ctx.register_model_graph(mg)
                return mg

            # Case 3: loaded graph differs + no overwrite -> throw error
            msg = (
                "A ModelGraph is already defined in the active ExperimentContext. "
                "Set `overwrite=True` to replace it or create a new context to "
                "load the new ModelGraph into."
            )
            raise RuntimeError(msg)

        exp_ctx.register_model_graph(mg)
        return mg
