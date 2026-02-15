from __future__ import annotations

import uuid
from typing import Any

from modularml.core.data.schema_constants import INVALID_LABEL_CHARACTERS
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.references.experiment_reference import ExperimentNodeReference
from modularml.utils.representation.summary import Summarizable


def generate_node_id() -> str:
    return str(uuid.uuid4())


class ExperimentNode(Summarizable, Configurable, Stateful):
    """
    Base class for all nodes within an Experiment.

    Each node is identified by a unique `label`.
    """

    def __init__(
        self,
        label: str,
        *,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a ExperimentNode with a name and register.

        Args:
            label (str):
                Unique identifier for this node.
            node_id (str, optional):
                Used only for de-serialization.
            register (bool, optional):
                Used only for de-serialization.

        """
        self._node_id: str = node_id or generate_node_id()
        self._label: str = label

        # Register to context
        if register:
            ctx = ExperimentContext.get_active()
            ctx.register_experiment_node(self)

    def _validate_label(self, label: str):
        if any(ch in label for ch in INVALID_LABEL_CHARACTERS):
            msg = (
                f"The label contains invalid characters: `{label}`. "
                f"Label cannot contain any of: {list(INVALID_LABEL_CHARACTERS)}"
            )
            raise ValueError(msg)

    @property
    def node_id(self) -> str:
        """Immutable internal identifier."""
        return self._node_id

    @property
    def label(self) -> str:
        """Get or set the unique label for this node."""
        return self._label

    @label.setter
    def label(self, new_label: str):
        """Get or set the unique label for this node."""
        self._validate_label(label=new_label)

        # Check registry (if registered)
        exp_ctx = ExperimentContext.get_active()
        if exp_ctx.has_node(node_id=self.node_id):
            exp_ctx.update_node_label(
                node_id=self.node_id,
                new_label=new_label,
                check_label_collision=True,
            )

        self._label = new_label

    # ================================================
    # Referencing
    # ================================================
    def reference(self) -> ExperimentNodeReference:
        return ExperimentNodeReference(
            node_id=self.node_id,
            node_label=self.label,
        )

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        return [
            ("label", self.label),
            ("node_id", self.node_id),
        ]

    def __repr__(self):
        return f"ExperimentNode(label='{self.label}')"

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "node_id": self.node_id,
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        register: bool = True,
    ) -> ExperimentNode:
        return cls(register=register, **config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        return {"label": self.label}

    def set_state(self, state: dict[str, Any]):
        self.label = state["label"]
