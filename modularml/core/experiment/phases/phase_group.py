"""Grouping utilities for sequencing experiment phases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.phases.phase import ExperimentPhase, InputBinding
from modularml.core.experiment.phases.train_phase import (
    BatchSchedulingPolicy,
    TrainPhase,
)
from modularml.utils.data.formatting import ensure_list

if TYPE_CHECKING:
    from collections.abc import Iterable

    from modularml.core.experiment.callbacks.callback import Callback
    from modularml.core.topology.graph_node import GraphNode
    from modularml.core.training.applied_loss import AppliedLoss


class PhaseGroup:
    """
    A named collection of experiment phases or nested phase groups.

    Description:
        PhaseGroups provide logical organization of related phases (e.g., a training
        phase + validation + evaluation phases). They can be nested (a PhaseGroup
        can contain other PhaseGroups) and are used as templates for cross-validation.

    Attributes:
        label: Name for this group of phases.
        phases: List of phases (or nested PhaseGroups) in execution order.

    Example:
    ```python
        group = PhaseGroup("training_workflow")
        group.add_phase(train_phase)
        group.add_phase(eval_phase)
    ```

    """

    def __init__(
        self,
        label: str,
        phases: list[ExperimentPhase | PhaseGroup] | None = None,
    ):
        """
        Initialize a PhaseGroup.

        Args:
            label: Name for this group.
            phases: Optional initial list of phases or nested groups.

        """
        self.label = label
        self._data: list[ExperimentPhase | PhaseGroup] = []
        self._existing_labels: set[str] = set()

        # To enforce unique labels across nested groups
        self._existing_labels.add(self.label)

        # Add any provided phases/groups
        if phases is not None:
            self.add_items(ensure_list(phases))

    def __repr__(self):
        return f"PhaseGroup(label={self.label}, entries={len(self._data)})"

    def _summary_rows(self) -> list[tuple]:
        rows = [("label", self.label)]
        for i, entry in enumerate(self._data):
            rows.append((str(i), f"{entry!r}"))
        return rows

    # ================================================
    # Properties
    # ================================================
    @property
    def all(self) -> list[ExperimentPhase | PhaseGroup]:
        """
        All registered phases and phase groups in execution order.

        This returns the top-level structure (i.e., include phase groups).
        Use `.flatten()` to unravel all nested groups into a single list
        of phases, in execution order.
        """
        return list(self._data)

    @property
    def phases(self) -> dict[str, ExperimentPhase]:
        """
        Only the top-level registered phases, keyed by their labels.

        The returned dict does not encode execution order. Use `all()`
        to get all phases and phase groups as executed.
        """
        return {ph.label: ph for ph in self._data if isinstance(ph, ExperimentPhase)}

    @property
    def phase_groups(self) -> dict[str, PhaseGroup]:
        """
        Only the top-level registered phase groups, keyed by their labels.

        The returned dict does not encode execution order. Use `all()`
        to get all phases and phase groups as executed.
        """
        return {
            ph_grp.label: ph_grp
            for ph_grp in self._data
            if isinstance(ph_grp, PhaseGroup)
        }

    # ================================================
    # Accessors
    # ================================================
    def _resolve_item_index(
        self,
        val: int | str | ExperimentPhase | PhaseGroup,
    ) -> int:
        # Grab label if instance
        if isinstance(val, (ExperimentPhase, PhaseGroup)):
            val = val.label

        if isinstance(val, int):
            return val
        if isinstance(val, str):
            if val not in self._existing_labels:
                msg = (
                    f"No phase/group exists with label: {val}. "
                    f"Available: {self._existing_labels}."
                )
                raise KeyError(msg)
            valid = [i for i, x in enumerate(self._data) if x.label == val]
            if len(valid) > 1:
                msg = f"Multiple items found with label '{val}'."
                raise ValueError(msg)
            return valid[0]

        msg = f"Invalid phase / group identifier: {type(val)}."
        raise TypeError(msg)

    def __getitem__(self, key: int | str) -> ExperimentPhase | PhaseGroup:
        """
        Retrieve the registered phase or group by its position or label.

        Args:
            key (int | str):
                The positional index (int) or label (str) of the phase
                or group to return.

        """
        pos_idx = self._resolve_item_index(key)
        return self._data[pos_idx]

    def items(self) -> Iterable[tuple[str, ExperimentPhase | PhaseGroup]]:
        """
        A generator over label-phase/group pairs.

        The returned items are keyed by their unique labels.
        They are not returned in execution order.

        Yields:
            tuple[str, ExperimentPhase | PhaseGroup]:
                Tuple of phase/group and its label.

        """
        by_lbl = {x.label: x for x in self._data}
        yield from by_lbl.items()

    def flatten(self) -> list[ExperimentPhase]:
        """
        Flattens all registered phase groups.

        Description:
            All phase groups are flattened into a their underlying phases.
            The returned list represents the execution order of all
            individual experiment phases.

        Returns:
            list[ExperimentPhase]:
                All flattened phased in execution order.

        """
        flat: list[ExperimentPhase] = []
        for x in self._data:
            if isinstance(x, PhaseGroup):
                flat.extend(x.flatten())
            else:
                flat.append(x)
        return flat

    def get_phase(self, key: int | str) -> ExperimentPhase:
        """
        Retrieves a registered phase from this group.

        Args:
            key (str, int):
                The label or registered position of the phase to return.

        Returns:
            ExperimentPhase: The specified phase.

        """
        x = self.__getitem__(key=key)
        if not isinstance(x, ExperimentPhase):
            msg = (
                f"Result with key '{key}' is a {type(x).__name__}, not ExperimentPhase."
            )
            raise TypeError(msg)
        return x

    def get_train_phase(self, key: int | str) -> TrainPhase:
        """
        Retrieves a registered training phase from this group.

        Args:
            key (str, int):
                The label or registered position of the phase to return.

        Returns:
            TrainPhase: The specified phase.

        """
        x = self.__getitem__(key=key)
        if not isinstance(x, TrainPhase):
            msg = f"Result with key '{key}' is a {type(x).__name__}, not TrainPhase."
            raise TypeError(msg)
        return x

    def get_eval_phase(self, key: int | str) -> EvalPhase:
        """
        Retrieves a registered evaluation phase from this group.

        Args:
            key (str, int):
                The label or registered position of the phase to return.

        Returns:
            EvalPhase: The specified phase.

        """
        x = self.__getitem__(key=key)
        if not isinstance(x, EvalPhase):
            msg = f"Result with key '{key}' is a {type(x).__name__}, not EvalPhase."
            raise TypeError(msg)
        return x

    def get_group(self, key: int | str) -> PhaseGroup:
        """
        Retrieves a registered sub-roup from this group.

        Args:
            key (str, int):
                The label or registered position of the group to return.

        Returns:
            PhaseGroup: The specified sub-group.

        """
        x = self.__getitem__(key=key)
        if not isinstance(x, PhaseGroup):
            msg = f"No PhaseGroup exists with key '{key}'."
            raise ValueError(msg)  # noqa: TRY004
        return x

    # ================================================
    # Registration / De-registration
    # ================================================
    def add_item(self, item: ExperimentPhase | PhaseGroup):
        """
        Registers a phase or group to this collection.

        Args:
            item (ExperimentPhase | PhaseGroup):
                New phase or phase group to append. The item must have
                a unique label relative to its parent group.

        """
        if not isinstance(item, (ExperimentPhase, PhaseGroup)):
            msg = f"Expected ExperimentPhase or PhaseGroup, got {type(item)}."
            raise TypeError(msg)
        if item.label in self._existing_labels:
            msg = f"An item already exists with label '{item.label}'."
            raise ValueError(msg)

        self._data.append(item)
        self._existing_labels.add(item.label)

    def add_items(self, items: Iterable[ExperimentPhase | PhaseGroup]):
        """
        Register phases or groups to this collection.

        Args:
            items (Iterable[ExperimentPhase | PhaseGroup]):
                New phases or groups to append. All items must have
                a unique label relative to its parent group.

        """
        for x in ensure_list(items):
            self.add_item(x)

    def remove_item(
        self,
        item: int | str | ExperimentPhase | PhaseGroup,
    ):
        """
        Remove a phase or phase group from this collection.

        Description:
            Removes a registered phase by its positional index, its label, or its
            instance.

        Args:
            item (int | str | ExperimentPhase | PhaseGroup):
                Item position, label, or instance to remove.

        """
        pos_idx = self._resolve_item_index(item)

        # Remove label
        self._existing_labels.remove(self._data[pos_idx].label)

        # Delete element from list
        del self._data[pos_idx]

    def remove_items(
        self,
        items: list[int | str | ExperimentPhase | PhaseGroup],
    ):
        """
        Remove phases or phase groups from this collection.

        Description:
            Removes all specified items from this collection.

        Args:
            items (list[int | str | ExperimentPhase | PhaseGroup]):
                List of phase/group position, label, or instance to remove.

        """
        for x in ensure_list(items):
            self.remove_item(x)

    def clear(self) -> PhaseGroup:
        """
        Removes all phases and groups registered within this group.

        Returns:
            PhaseGroup: self

        """
        self._data.clear()
        self._existing_labels.clear()

        return self

    def add_phase(self, phase: ExperimentPhase) -> PhaseGroup:
        """
        Register a phase to this group.

        Description:
            Registers a new phase to be run wihtin this phase group.
            Phases are executed in the order they are added.

        Args:
            phase (ExperimentPhase):
                A fully constructed ExperimentPhase.

        Returns:
            PhaseGroup: self

        """
        if not isinstance(phase, ExperimentPhase):
            msg = f"Expected ExperimentPhase, got {type(phase)}."
            raise TypeError(msg)

        self.add_item(phase)

        return self

    def add_group(self, group: PhaseGroup) -> PhaseGroup:
        """
        Register a sub-group to this group.

        Description:
            Registers a new phase to be run wihtin this phase group.
            Phases are executed in the order they are added.

        Args:
            group (PhaseGroup):
                A fully constructed phase group to nest under this one.

        Returns:
            PhaseGroup: self

        """
        if not isinstance(group, PhaseGroup):
            msg = f"Expected PhaseGroup, got {type(group)}."
            raise TypeError(msg)

        self.add_item(group)

        return self

    def remove_phase(self, phase: int | str | ExperimentPhase) -> PhaseGroup:
        """
        Remove a phase from this group.

        Description:
            Removes a registered phase by index, label, or instance.

        Args:
            phase (str | int | ExperimentPhase):
                Phase position, label, or instance to remove.

        Returns:
            PhaseGroup: self

        """
        self.remove_item(phase)
        return self

    def remove_group(self, group: int | str | PhaseGroup) -> PhaseGroup:
        """
        Remove a sub-group from this group.

        Description:
            Removes a registered group by index, label, or instance.

        Args:
            group (str | int | PhaseGroup):
                Phase position, label, or instance to remove.

        Returns:
            PhaseGroup: self

        """
        self.remove_item(group)
        return self

    # ================================================
    # TrainPhase
    # ================================================
    def add_train_phase(
        self,
        label: str,
        *,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss],
        n_epochs: int = 1,
        active_nodes: list[GraphNode] | None = None,
        batch_schedule: BatchSchedulingPolicy | str = BatchSchedulingPolicy.ZIP_STRICT,
        callbacks: list[Callback] | None = None,
    ) -> PhaseGroup:
        """
        Constructs and registers a new training phase.

        Args:
            label (str):
                A label to assign to this training phase. Must be unique to already
                registered phases.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph.

            losses (list[AppliedLoss]):
                A list of losses to be applied during this training phase.

            n_epochs (int):
                Number of epochs to perform.

            active_nodes (list[str  |  GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            batch_schedule (str | BatchSchedulingPolicy, optional):
                Defines how batches from multiple samplers are scheduled during training.
                This is only relevant if more than one sampler is defined in
                `input_sources`.

                Let samplers `S1` and `S2` produce: `S1 = [b1, b2, b3]` and  `S2 = [c1, c2]`

                The outputs of each policy is given below:

                ```python
                "zip_strict": (b1, c1), (b2, c2)
                "zip_cycle": (b1, c1), (b2, c2), (b3, c1)
                "alternate_strict": b1, c1, b2, c2
                "alternate_cycle": b1, c1, b2, c2, b3, c1
                ```

                See also :class:`BatchSchedulingPolicy`.

            callbacks (list[Callback] | None, optional):
                An optional list of Callbacks to run during phase execution.

        Returns:
            PhaseGroup: self

        """
        phase = TrainPhase(
            label=label,
            input_sources=input_sources,
            losses=losses,
            n_epochs=n_epochs,
            active_nodes=active_nodes,
            batch_schedule=batch_schedule,
            callbacks=callbacks,
        )
        return self.add_phase(phase=phase)

    add_training = add_train_phase
    add_train = add_train_phase

    # ================================================
    # EvalPhase
    # ================================================
    def add_eval_phase(
        self,
        label: str,
        *,
        input_sources: list[InputBinding],
        losses: list[AppliedLoss] | None = None,
        active_nodes: list[GraphNode] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> PhaseGroup:
        """
        Constructs and registers a new evaluation phase.

        Args:
            label (str):
                A label to assign to this evaluation phase. Must be unique to already
                registered phases.

            input_sources (list[InputBinding]):
                Input bindings for each head node in ModelGraph.

            losses (list[AppliedLoss]):
                An optional list of losses to be applied during evaluation.

            active_nodes (list[GraphNode] | None, optional):
                A list of active GraphNodes in this phase of the experiment. Nodes can
                be listed via their ID, label, or with the actual node instance. If
                None, all nodes comprising the ModelGraph are used. Defaults to None.

            callbacks (list[Callback] | None, optional):
                An optional list of Callbacks to run during phase execution.

        Returns:
            PhaseGroup: self

        """
        phase = EvalPhase(
            label=label,
            input_sources=input_sources,
            losses=losses,
            active_nodes=active_nodes,
            callbacks=callbacks,
        )
        return self.add_phase(phase=phase)

    add_evaluation = add_eval_phase
    add_eval = add_eval_phase

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration details required to reconstruct this phase group.

        Returns:
            dict[str, Any]:
                Configuration used to reconstruct the phase group.

        """
        items_cfg = []
        for entry in self._data:
            if isinstance(entry, PhaseGroup):
                items_cfg.append(
                    {
                        "item_type": "PhaseGroup",
                        "config": entry.get_config(),
                    },
                )
            elif isinstance(entry, ExperimentPhase):
                items_cfg.append(
                    {
                        "item_type": "ExperimentPhase",
                        "config": entry.get_config(),
                    },
                )
            else:
                msg = f"Unexpected item type in PhaseGroup: {type(entry)}."
                raise TypeError(msg)
        return {"label": self.label, "items": items_cfg}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PhaseGroup:
        """
        Construct a phase group from a configuration dictionary.

        Args:
            config (dict[str, Any]):
                Configuration details. Keys must be strings.

        Returns:
            PhaseGroup: Reconstructed phase group.

        """
        items = []
        for item_cfg in config.get("items", []):
            item_type = item_cfg["item_type"]
            inner_cfg = item_cfg["config"]
            if item_type == "PhaseGroup":
                items.append(cls.from_config(inner_cfg))
            elif item_type == "ExperimentPhase":
                items.append(ExperimentPhase.from_config(inner_cfg))
            else:
                msg = f"Unknown item_type in PhaseGroup config: {item_type}."
                raise ValueError(msg)
        return cls(label=config["label"], phases=items)
