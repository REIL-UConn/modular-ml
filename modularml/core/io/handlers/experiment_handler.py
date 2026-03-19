"""Handler for serializing :class:`Experiment` objects and dependencies."""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.data.featureset import FeatureSet
from modularml.core.io.handlers.handler import BaseHandler
from modularml.core.topology.model_graph import ModelGraph
from modularml.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.io.serializer import LoadContext, SaveContext

logger = get_logger(level=logging.INFO)


class ExperimentHandler(BaseHandler["Experiment"]):
    """
    Serialize :class:`Experiment` instances and their dependencies.

    Attributes:
        object_version (str): Semantic version of emitted artifacts.
        config_rel_path (str): Relative JSON filename for configs.
        state_rel_path (str): Relative pickle filename for runtime state.

    """

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # Encoding
    # ================================================
    def encode_state(
        self,
        obj: Experiment,
        save_dir: Path,
        *,
        ctx: SaveContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Encode :class:`Experiment` dependencies and runtime state.

        Description:
            FeatureSets, :class:`ModelGraph`, and checkpoints are saved as nested
            artifacts alongside pickled runtime metadata.

        Args:
            obj (Experiment): Experiment instance being serialized.
            save_dir (Path): Destination directory for artifact content.
            ctx (SaveContext): Active :class:`SaveContext`.

        Returns:
            dict[str, Any]: Mapping of logical artifact keys to file locations.

        """
        save_dir = Path(save_dir)
        file_mapping: dict[str, Any] = {}

        # ------------------------------------------------
        # 1. Save FeatureSets as sub-artifacts
        # ------------------------------------------------
        featuresets = obj.ctx.available_featuresets
        if featuresets:
            fs_dir = save_dir / "featuresets"
            fs_dir.mkdir(exist_ok=True)

            # Serialize each featureset
            fs_entries: dict[str, str] = {}
            for fs_id, fs in featuresets.items():
                save_path = fs.save(filepath=(fs_dir / fs_id), overwrite=True)
                fs_entries[fs_id] = str(Path(save_path).relative_to(save_dir))

            # Mapping of each featureset to its serialized file
            file_mapping["featuresets"] = fs_entries

        # ------------------------------------------------
        # 2. Save ModelGraph as sub-artifact
        # ------------------------------------------------
        mg = obj.model_graph
        if mg is not None:
            mg_dir = save_dir / "model_graph"
            mg_dir.mkdir(exist_ok=True)

            # Serialize model graph
            save_path = mg.save(filepath=(mg_dir / mg.label), overwrite=True)
            file_mapping["model_graph"] = str(Path(save_path).relative_to(save_dir))

        # ------------------------------------------------
        # 3. Copy on-disk checkpoints into the artifact
        # ------------------------------------------------
        ckpt_entries: dict[str, str] = {}
        for ckpt_label, ckpt_src in obj._checkpoints.items():
            ckpt_path = Path(ckpt_src)
            if not ckpt_path.exists():
                continue

            # Destination preserves the label's nested structure
            # e.g. "training/epoch_0_ckpt" -> checkpoints/training/epoch_0_ckpt.ckpt.mml
            dest = save_dir / "checkpoints" / ckpt_label
            # Ensure the suffix is carried over
            if ckpt_path.suffix != dest.suffix:
                dest = dest.with_name(dest.name + "".join(ckpt_path.suffixes))
            dest.parent.mkdir(parents=True, exist_ok=True)

            if ckpt_path.is_dir():
                shutil.copytree(ckpt_path, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(ckpt_path, dest)

            ckpt_entries[ckpt_label] = str(dest.relative_to(save_dir))

        if ckpt_entries:
            file_mapping["checkpoints"] = ckpt_entries

        # ------------------------------------------------
        # 4. Copy disk-backed result stores into the artifact
        # ------------------------------------------------
        results_manifest = self._collect_result_files(obj, save_dir)
        if results_manifest:
            file_mapping["results"] = results_manifest

        # ------------------------------------------------
        # 5. Save experiment-specific runtime state
        # ------------------------------------------------
        exp_state = {
            "history": obj._history,
            "mg_state": mg.get_state(),
        }
        state_path = save_dir / self.state_rel_path
        with Path.open(state_path, "wb") as f:
            pickle.dump(exp_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_mapping["state"] = self.state_rel_path

        return file_mapping

    def _collect_result_files(
        self,
        obj: Experiment,
        save_dir: Path,
    ) -> dict[str, Any]:
        """Copy disk-backed result stores into the archive and return the manifest."""
        from modularml.core.experiment.results.group_results import PhaseGroupResults

        manifest: dict[str, dict] = {}
        for run_idx, run in enumerate(obj._history):
            if run.results is None:
                continue

            if isinstance(run.results, PhaseGroupResults):
                phases = run.results.flatten()
            else:
                phases = {run.results.label: run.results}

            for label, pr in phases.items():
                entry: dict[str, Any] = {}
                base = save_dir / "results" / str(run_idx) / label

                exec_loc = pr._execution._location
                if exec_loc is not None and Path(exec_loc).is_dir():
                    dest = base / "execution_data"
                    shutil.copytree(exec_loc, dest, dirs_exist_ok=True)
                    entry["execution_data"] = str(dest.relative_to(save_dir))

                met_loc = pr._metrics._location
                if met_loc is not None and Path(met_loc).is_dir():
                    dest = base / "metrics"
                    shutil.copytree(met_loc, dest, dirs_exist_ok=True)
                    entry["metrics"] = str(dest.relative_to(save_dir))

                art_loc = pr._artifacts._location
                if art_loc is not None and Path(art_loc).is_dir():
                    dest = base / "artifacts"
                    shutil.copytree(art_loc, dest, dirs_exist_ok=True)
                    entry["artifacts"] = str(dest.relative_to(save_dir))

                # Collect callback sub-phase results (e.g. Evaluation callbacks)
                cb_entries = self._collect_callback_result_files(
                    pr=pr,
                    base=base / "callbacks",
                    save_dir=save_dir,
                )
                if cb_entries:
                    entry["callbacks"] = cb_entries

                if entry:
                    manifest.setdefault(str(run_idx), {})[label] = entry

        return manifest

    def _collect_callback_result_files(
        self,
        pr: Any,
        base: Path,
        save_dir: Path,
    ) -> dict[str, Any]:
        """Copy disk-backed stores from callback sub-phases and return their manifest."""
        from modularml.callbacks.evaluation import EvaluationCallbackResult

        cb_manifest: dict[str, Any] = {}
        for cb_res in pr._callbacks:
            if not isinstance(cb_res, EvaluationCallbackResult):
                continue
            eval_res = cb_res.eval_results
            if eval_res is None:
                continue

            cb_label = cb_res.callback_label
            cb_entry: dict[str, str] = {}
            cb_base = base / cb_label

            exec_loc = eval_res._execution._location
            if exec_loc is not None and Path(exec_loc).is_dir():
                dest = cb_base / "execution_data"
                shutil.copytree(exec_loc, dest, dirs_exist_ok=True)
                cb_entry["execution_data"] = str(dest.relative_to(save_dir))

            met_loc = eval_res._metrics._location
            if met_loc is not None and Path(met_loc).is_dir():
                dest = cb_base / "metrics"
                shutil.copytree(met_loc, dest, dirs_exist_ok=True)
                cb_entry["metrics"] = str(dest.relative_to(save_dir))

            art_loc = eval_res._artifacts._location
            if art_loc is not None and Path(art_loc).is_dir():
                dest = cb_base / "artifacts"
                shutil.copytree(art_loc, dest, dirs_exist_ok=True)
                cb_entry["artifacts"] = str(dest.relative_to(save_dir))

            if cb_entry:
                cb_manifest[cb_label] = cb_entry

        return cb_manifest

    # ================================================
    # Decoding
    # ================================================
    def decode(
        self,
        cls: type[Experiment],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> Experiment:
        """
        Rebuild an :class:`Experiment` and its dependencies from disk.

        The active :class:`~modularml.core.experiment.experiment_context.ExperimentContext`
        must not already contain an :class:`Experiment` or :class:`ModelGraph`.

        Args:
            cls (type[Experiment]): Experiment class to reconstruct.
            load_dir (Path): Directory containing the saved artifact.
            ctx (LoadContext): Active :class:`LoadContext`.

        Returns:
            Experiment: Reconstructed experiment bound to the active context.

        Raises:
            RuntimeError: If the active context already contains conflicting artifacts.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext

        load_dir = Path(load_dir)
        exp_ctx = ExperimentContext.get_active()

        # ------------------------------------------------
        # Validate that the context is clean
        # ------------------------------------------------
        if exp_ctx.get_experiment() is not None:
            if not ctx.overwrite_collision:
                msg = (
                    "Cannot load Experiment into the active ExperimentContext: "
                    "an Experiment is already associated with it. Create a new "
                    "ExperimentContext before loading."
                )
                raise RuntimeError(msg)

            # Creates a new, empty context
            exp_ctx = ExperimentContext()

        if exp_ctx.model_graph is not None:
            msg = (
                "Cannot load Experiment into the active ExperimentContext: "
                "a ModelGraph is already registered. Create a new "
                "ExperimentContext before loading."
            )
            raise RuntimeError(msg)

        # Read file mapping from the experiment's artifact.json
        artifact_data = self._read_json(load_dir / "artifact.json")
        file_mapping: dict[str, Any] = artifact_data["files"]

        # ------------------------------------------------
        # 1. Load FeatureSets
        # ------------------------------------------------
        fs_entries: dict[str, str] = file_mapping.get("featuresets", {})
        for fs_rel_path in fs_entries.values():
            # Get filepath to serialized object (.fs.mml file)
            file_fs = load_dir / fs_rel_path

            # Load featureset - this automatically registers to the active ctx
            _ = FeatureSet.load(
                filepath=file_fs,
                allow_packaged_code=ctx.allow_packaged_code,
                overwrite=ctx.overwrite_collision,
            )

        # ------------------------------------------------
        # 2. Load ModelGraph (registers nodes + graph)
        # ------------------------------------------------
        mg_rel_path = file_mapping.get("model_graph")
        if mg_rel_path is not None:
            # Get filepath to serialized object (.mg.mml file)
            file_mg = load_dir / mg_rel_path

            # Load ModelGraph - handles node and graph registration
            _ = ModelGraph.load(
                filepath=file_mg,
                allow_packaged_code=ctx.allow_packaged_code,
                overwrite=ctx.overwrite_collision,
            )

        # ------------------------------------------------
        # 3. Reconstruct Experiment from config
        # ------------------------------------------------
        json_cfg = self.decode_config(load_dir=load_dir, ctx=ctx)
        cfg = self._restore_json_cfg(data=json_cfg, ctx=ctx)
        exp = cls.from_config(cfg)

        # ------------------------------------------------
        # 4. Restore experiment runtime state
        # ------------------------------------------------
        exp_state: dict[str, Any] = self.decode_state(
            load_dir=load_dir,
            ctx=ctx,
        )
        exp._history = exp_state.get("history", [])

        # ------------------------------------------------
        # 5. Restore disk-backed result stores
        # ------------------------------------------------
        results_manifest: dict[str, Any] = file_mapping.get("results", {})
        if results_manifest:
            results_dir: Path | None = ctx.extras.get("results_dir")
            if results_dir is not None:
                self._restore_result_files(
                    exp=exp,
                    load_dir=load_dir,
                    results_dir=Path(results_dir),
                    manifest=results_manifest,
                )
            else:
                from modularml.utils.logging.warnings import warn

                warn(
                    "The serialized experiment contains disk-backed result stores "
                    "(execution contexts or artifacts), but no `results_dir` was "
                    "provided to `Experiment.load()`. These results were not restored.",
                    hints=[
                        "Pass `results_dir=Path(...)` to `Experiment.load()` "
                        "to extract and re-link disk-backed result stores.",
                    ],
                    stacklevel=2,
                )

        # ------------------------------------------------
        # 6. Extract checkpoints to user-provided directory
        # ------------------------------------------------
        ckpt_entries: dict[str, str] = file_mapping.get("checkpoints", {})
        checkpoint_dir: Path | None = ctx.extras.get("checkpoint_dir")

        if ckpt_entries and checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            exp.set_checkpoint_dir(checkpoint_dir, create=False)

            for label, rel_path in ckpt_entries.items():
                src_path = load_dir / rel_path
                if not src_path.exists():
                    continue

                # Preserve nested structure from the label
                dest = checkpoint_dir / label
                if src_path.suffix != dest.suffix:
                    dest = dest.with_name(
                        dest.name + "".join(src_path.suffixes),
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)

                if src_path.is_dir():
                    shutil.copytree(src_path, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest)

                exp._checkpoints[label] = dest

        elif ckpt_entries:
            from modularml.utils.logging.warnings import warn

            warn(
                f"The serialized experiment contains "
                f"{len(ckpt_entries)} checkpoint(s), but no "
                f"`checkpoint_dir` was provided to "
                f"`Experiment.load()`. Checkpoints were not "
                f"restored.",
                hints=[
                    "Pass `checkpoint_dir=Path(...)` to "
                    "`Experiment.load()` to extract checkpoints.",
                ],
                stacklevel=2,
            )

        return exp

    def _restore_result_files(
        self,
        exp: Experiment,
        load_dir: Path,
        results_dir: Path,
        manifest: dict[str, Any],
    ) -> None:
        """Extract archived result files and re-link stores in ``exp._history``."""
        from modularml.core.experiment.results.artifact_store import ArtifactStore
        from modularml.core.experiment.results.execution_store import ExecutionStore
        from modularml.core.experiment.results.group_results import PhaseGroupResults
        from modularml.core.experiment.results.metric_store import MetricStore

        results_dir.mkdir(parents=True, exist_ok=True)

        for run_idx, run in enumerate(exp._history):
            run_manifest: dict[str, Any] = manifest.get(str(run_idx), {})
            if not run_manifest or run.results is None:
                continue

            if isinstance(run.results, PhaseGroupResults):
                phases = run.results.flatten()
            else:
                phases = {run.results.label: run.results}

            for label, pr in phases.items():
                phase_manifest: dict[str, Any] = run_manifest.get(label, {})
                if not phase_manifest:
                    continue

                base = results_dir / str(run_idx) / label

                exec_rel = phase_manifest.get("execution_data")
                if exec_rel is not None:
                    src = load_dir / exec_rel
                    dest = base / "execution_data"
                    if src.is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                        pr._execution = ExecutionStore.from_directory(dest)
                        pr._execution_dir = dest

                met_rel = phase_manifest.get("metrics")
                if met_rel is not None:
                    src = load_dir / met_rel
                    dest = base / "metrics"
                    if src.is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                        pr._metrics = MetricStore.from_directory(dest)
                        pr._metric_dir = dest

                art_rel = phase_manifest.get("artifacts")
                if art_rel is not None:
                    src = load_dir / art_rel
                    dest = base / "artifacts"
                    if src.is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                        pr._artifacts = ArtifactStore.from_directory(dest)
                        pr._artifact_dir = dest

                # Restore callback sub-phase stores
                cb_manifest: dict[str, Any] = phase_manifest.get("callbacks", {})
                if cb_manifest:
                    self._restore_callback_result_files(
                        pr=pr,
                        load_dir=load_dir,
                        base=base / "callbacks",
                        cb_manifest=cb_manifest,
                        artifact_store_cls=ArtifactStore,
                        execution_store_cls=ExecutionStore,
                        metric_store_cls=MetricStore,
                    )

                pr._series_cache.clear()

    def _restore_callback_result_files(
        self,
        pr: Any,
        load_dir: Path,
        base: Path,
        cb_manifest: dict[str, Any],
        *,
        artifact_store_cls: type,
        execution_store_cls: type,
        metric_store_cls: type,
    ) -> None:
        """Re-link disk-backed stores for callback sub-phase results."""
        from modularml.callbacks.evaluation import EvaluationCallbackResult

        for cb_res in pr._callbacks:
            if not isinstance(cb_res, EvaluationCallbackResult):
                continue
            cb_label = cb_res.callback_label
            cb_entry: dict[str, str] = cb_manifest.get(cb_label, {})
            if not cb_entry or cb_res.eval_results is None:
                continue

            eval_res = cb_res.eval_results
            cb_base = base / cb_label

            exec_rel = cb_entry.get("execution_data")
            if exec_rel is not None:
                src = load_dir / exec_rel
                dest = cb_base / "execution_data"
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                    eval_res._execution = execution_store_cls.from_directory(dest)
                    eval_res._execution_dir = dest

            met_rel = cb_entry.get("metrics")
            if met_rel is not None:
                src = load_dir / met_rel
                dest = cb_base / "metrics"
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                    eval_res._metrics = metric_store_cls.from_directory(dest)
                    eval_res._metric_dir = dest

            art_rel = cb_entry.get("artifacts")
            if art_rel is not None:
                src = load_dir / art_rel
                dest = cb_base / "artifacts"
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                    eval_res._artifacts = artifact_store_cls.from_directory(dest)
                    eval_res._artifact_dir = dest

            eval_res._series_cache.clear()
