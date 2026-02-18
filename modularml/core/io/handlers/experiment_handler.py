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
    """Handler for Experiment objects."""

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
        Encode Experiment state into save_dir.

        FeatureSets and ModelGraph are saved as full sub-artifacts (each
        with its own `artifact.json`, `config.json`, `state.pkl`).
        The experiment's own runtime state (history, checkpoints) is
        pickled separately.

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
        # 4. Save experiment-specific runtime state
        # ------------------------------------------------
        exp_state = {
            "history": obj._history,
        }
        state_path = save_dir / self.state_rel_path
        with Path.open(state_path, "wb") as f:
            pickle.dump(exp_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_mapping["state"] = self.state_rel_path

        return file_mapping

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
        Decode an Experiment from a saved artifact directory.

        The active `ExperimentContext` must be clean (no existing
        Experiment or ModelGraph). FeatureSets and ModelGraph are loaded
        first so that the execution plan can reference registered nodes.

        """
        from modularml.core.experiment.experiment_context import ExperimentContext

        load_dir = Path(load_dir)
        exp_ctx = ExperimentContext.get_active()

        # ------------------------------------------------
        # Validate that the context is clean
        # ------------------------------------------------
        if exp_ctx.get_experiment() is not None:
            msg = (
                "Cannot load Experiment into the active ExperimentContext: "
                "an Experiment is already associated with it. Create a new "
                "ExperimentContext before loading."
            )
            raise RuntimeError(msg)

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
        # 5. Extract checkpoints to user-provided directory
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
