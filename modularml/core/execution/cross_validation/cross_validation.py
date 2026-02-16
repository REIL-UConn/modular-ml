from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.execution.cross_validation.cv_binding import CVBinding, _FoldViews
from modularml.core.execution.strategy import ExecutionStrategy
from modularml.core.experiment.experiment import Experiment
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.experiment.phases.phase_group import PhaseGroup
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.splitters.random_splitter import RandomSplitter
from modularml.utils.data.formatting import ensure_list
from modularml.utils.environment.environment import IN_NOTEBOOK
from modularml.utils.progress_bars.progress_task import ProgressTask

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.execution.cross_validation.cv_results import CVResults


class CrossValidation(ExecutionStrategy):
    def __init__(
        self,
        *,
        label: str = "CV",
        bindings: CVBinding | list[CVBinding],
        n_folds: int = 5,
        seed: int = 13,
        phase: TrainPhase | PhaseGroup | None = None,
        experiment: Experiment | None = None,
    ):
        self.label = label
        self.seed = int(seed)

        # Get experiment on which CV is applied
        if experiment is None:
            experiment = ExperimentContext.get_active().get_experiment()
        if not isinstance(experiment, Experiment):
            msg = (
                "Cross validation requires a reference to an experiment. "
                "Either provide one explicitly, or set the active context."
            )
            raise TypeError(msg)
        self.experiment = experiment

        # Validate template phase group to perform CV over
        self.phase_template = PhaseGroup(label=self.label)
        if phase is not None:
            if isinstance(phase, TrainPhase):
                self.phase_template.add_phase(phase=phase)
            elif isinstance(phase, PhaseGroup):
                self.phase_template.add_group(group=phase)
            else:
                msg = f"Expected TrainPhase or PhaseGroup. Received: {type(phase)}."
                raise TypeError(msg)
        # If no phases given, use all defined in experiment
        else:
            self.phase_template = self.experiment.execution_plan
            self.phase_template.label = self.label

        # Validate bindings
        self.bindings: dict[str, CVBinding] = {
            b._fs_id: b for b in ensure_list(bindings)
        }
        for b in self.bindings.values():
            if not isinstance(b, CVBinding):
                msg = f"Expected CVBinding. Received: {type(b)}."
                raise TypeError(msg)

        # Validate `n_folds` and `val_size`
        if (not isinstance(n_folds, int)) or (n_folds < 1):
            msg = "`n_folds` must be an integer greater than or equal to 1."
            raise ValueError(msg)
        for b in self.bindings.values():
            if (b.val_size is not None) and (b.val_size > (1 / n_folds)):
                msg = (
                    "`val_size` cannot be larger than `1/n_folds`: "
                    f"{b.val_size} > {1 / n_folds}."
                )
                raise ValueError(msg)
        self.n_folds = n_folds

    def _generate_fold_views(self) -> dict[int, dict[str, _FoldViews]]:
        """
        Generates views of each fold's train and val splits.

        Returns:
            dict[int, dict[str, _FoldViews]]:
                Fold-specific views keyed by fold_idx, then by FeatureSet node id.

        """
        # Precompute all fold views (keyed by fold and FeatureSet.node_id)
        all_folds: dict[int, dict[str, _FoldViews]] = defaultdict(dict)

        for fs_id, cv_binding in self.bindings.items():
            # Get referenced FeatureSet
            fs: FeatureSet = self.experiment.ctx.get_node(
                node_id=fs_id,
                enforce_type="FeatureSet",
            )

            # Build view over all splits in this binding
            views = [fs.get_split(spl) for spl in cv_binding.source_splits]
            unique_rows = np.unique(np.hstack([v.indices for v in views]))
            src_view = FeatureSetView.from_featureset(
                fs=fs,
                rows=unique_rows,
                label="cv_pool",
            )

            # Construct splitter
            if cv_binding.val_size is None:
                fold_ratios = {
                    f"fold_{i}": 1 / self.n_folds for i in range(self.n_folds)
                }
            else:
                fold_ratios = {
                    f"fold_{i}": cv_binding.val_size for i in range(self.n_folds)
                }
                fold_ratios["remaining"] = 1 - self.n_folds * cv_binding.val_size
            cv_splitter = RandomSplitter(
                ratios=fold_ratios,
                stratify_by=cv_binding.stratify_by,
                group_by=cv_binding.group_by,
                seed=self.seed,
            )

            # Create and record folds
            fold_splits: dict[str, FeatureSetView] = cv_splitter.split(
                view=src_view,
                return_views=True,
            )
            for i in range(self.n_folds):
                val_view = fold_splits[f"fold_{i}"]
                val_view.label = "val"
                train_view = src_view.take_difference(val_view, label="train")

                fold_views = _FoldViews(
                    fold_idx=i,
                    train=train_view,
                    val=val_view,
                )
                if fs_id in all_folds[i]:
                    msg = (
                        f"Data for fold {i} already exists for FeatureSet '{fs.label}'."
                    )
                    raise KeyError(msg)
                all_folds[i][fs_id] = fold_views

        return all_folds

    def _replace_featuresets(
        self,
        fold_views: dict[str, _FoldViews],
        ctx: ExperimentContext,
    ):
        """
        Updates all featuresets in the given context to use the fold splits.

        Args:
            fold_views (dict[str, _FoldViews]):
                Fold splits to use, keyed by FeatureSet.node_id

            ctx (ExperimentContext):
                The ExperimentContext in which to update FeatureSets.

        """
        for fs_id, fold_data in fold_views.items():
            # Get old FeatureSet and remove from ctx
            fs_old: FeatureSet = ctx.remove_node(
                node_id=fs_id,
                error_if_missing=True,
            )

            # Register new FeatureSet w/o splits or scalers
            fs_new = fs_old.copy(
                label=fs_old.label,
                share_raw_data_buffer=True,
                restore_splits=False,
                restore_scalers=False,
                register=False,
            )
            fs_new._node_id = fs_old.node_id
            ctx.register_experiment_node(
                node=fs_new,
                check_label_collision=True,
            )

            # Since splits not involved in CV may be needed
            # We must walk back over the recorded split recs
            # Only the "train" and "eval" splits will be overwritten
            cv_binding = self.bindings[fs_id]
            for rec in sorted(fs_old._split_recs, key=lambda r: r.order):
                # If rec is applied to the "train" or "eval", we need to re-execute
                if rec.applied_to.split_name in [
                    cv_binding.train_split_name,
                    cv_binding.val_split_name,
                ]:
                    src_to_split = fs_new.get_split(
                        split_name=rec.applied_to.split_name,
                    )
                    src_to_split.split(
                        splitter=rec.splitter,
                        return_views=False,
                        register=True,
                    )

                # Otherwise we can directly copy sample IDs
                else:
                    for spl_name in rec.produced_splits:
                        # Check if uses fold specific views
                        if spl_name == cv_binding.train_split_name:
                            new_view = fold_data.train
                        elif spl_name == cv_binding.val_split_name:
                            new_view = fold_data.val
                        else:
                            new_view = fs_old.get_split(split_name=spl_name)

                        # Copy exact view (exact sample IDs)
                        fs_new._splits[spl_name] = FeatureSetView.from_featureset(
                            fs=fs_new,
                            rows=new_view.indices,
                            columns=new_view.columns,
                            label=spl_name,
                        )

                    # Ensure record is attached to new fs
                    fs_new._split_recs.append(rec)

            # Reapply scalers
            for rec in sorted(fs_old._scaler_recs, key=lambda r: r.order):
                fs_new.fit_transform(
                    scaler=rec.scaler_obj,
                    domain=rec.domain,
                    keys=rec.keys,
                    fit_to_split=rec.fit_split,
                    merged_axes=rec.merged_axes,
                )

    def run(
        self,
        *,
        show_fold_progress: bool = True,
        persist_progress: bool = IN_NOTEBOOK,
        **kwargs,
    ) -> CVResults:
        """
        Execute cross-validation over all folds.

        Args:
            show_fold_progress (bool, optional):
                Whether to show a progress bar over fold execution.
                Defaults to True.
            persist_progress (bool, optional):
                Whether to leave progress bars (including nested ones) shown after
                they complete. Defaults to `IN_NOTEBOOK` (True if working in a
                notebook, False if in a Python script).
            **kwargs:
                Display flags forwarded to the phase-specific run method.

        Returns:
            CVResults: Results for all folds.

        """
        from modularml.core.execution.cross_validation.cv_results import CVResults

        # Precompute all train/val FeatureSetViews for each fold
        all_folds = self._generate_fold_views()

        # ------------------------------------------------
        # Progress Bar: folds
        # ------------------------------------------------
        fold_ptask = ProgressTask(
            style="cross_validation",
            description=f"Cross-Validation ['{self.label}']",
            total=self.n_folds,
            enabled=show_fold_progress,
            persist=persist_progress,
        )
        fold_ptask.start()
        kwargs["persist_progress"] = persist_progress

        # ------------------------------------------------
        # Fold Execution
        # ------------------------------------------------
        all_res = CVResults(label=self.label)
        for fold_idx in range(self.n_folds):
            # Create temporary context and replace featureset
            with self.experiment.ctx.temporary() as tmp_ctx:
                # Update all featuresets participating in CV
                self._replace_featuresets(
                    fold_views=all_folds[fold_idx],
                    ctx=tmp_ctx,
                )

                # Run `self.phase_template`
                fold_res = self.experiment.run_phase_group(
                    group=self.phase_template,
                    **kwargs,
                )

                # Record fold results
                fold_res.label = f"fold_{fold_idx}"
                all_res.add_result(result=fold_res)

            fold_ptask.tick(n=1)

        fold_ptask.finish()

        return all_res
