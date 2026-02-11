from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from modularml.core.data.featureset import FeatureSet
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    REP_TRANSFORMED,
)
from modularml.core.io.handlers.handler import BaseHandler
from modularml.core.io.protocols import Stateful
from modularml.utils.data.pyarrow_data import hash_pyarrow_table
from modularml.utils.logging import get_logger

if TYPE_CHECKING:
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_collection import SampleCollection
    from modularml.core.io.serializer import LoadContext, SaveContext
    from modularml.core.splitting.splitter_record import SplitterRecord
    from modularml.core.transforms.scaler_record import ScalerRecord


logger = get_logger(level=logging.INFO)


class FeatureSetHandler(BaseHandler[FeatureSet]):
    """BaseHandler for FeatureSet objects."""

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # FeatureSet Encoding
    # ================================================
    def encode_state(
        self,
        obj: FeatureSet,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (FeatureSet):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        if not isinstance(obj, Stateful):
            raise NotImplementedError(
                "FeatureSet must implement the `Stateful` protocol.",
            )
        save_dir = Path(save_dir)

        # Track which files go where
        file_mapping: dict[str, str] = {}

        # Grab FeatureSet state (holds live instances)
        state = obj.get_state()

        # ------------------------------------------------
        # Save SampleCollection
        #  - uses PyArrow file to store data + schema
        # ------------------------------------------------
        coll: SampleCollection = state.pop("sample_collection")
        coll_path = coll.save(Path(save_dir) / "sample_collection.arrow")
        file_mapping["sample_collection"] = coll_path.name

        # ------------------------------------------------
        # Save Split Configs
        #  - removes source reference
        # ------------------------------------------------
        split_cfgs = {}
        splits: dict[str, FeatureSetView] = state.pop("splits")
        for k, fsv in splits.items():
            split_cfgs[k] = {
                "indices": fsv.indices.tolist()
                if isinstance(fsv.indices, np.ndarray)
                else fsv.indices,
                "columns": fsv.columns,
                "label": fsv.label,
            }
        split_cfg_path = self._write_json(
            data=split_cfgs,
            save_path=Path(save_dir) / "split_configs.json",
        )
        file_mapping["split_configs"] = split_cfg_path.name

        # ------------------------------------------------
        # Save ScalerRecords
        #  - only configs used during reload (refit)
        #  - full artifact saved for tracking only
        # ------------------------------------------------
        dir_scaler_rec = Path(save_dir) / "scaler_records"
        dir_scaler_rec.mkdir(exist_ok=True)

        scaler_cfg_files: list[str] = []
        scaler_recs: list[ScalerRecord] = sorted(
            state.pop("scaler_records"),
            key=lambda x: x.order,
        )
        n_digits = (len(scaler_recs) // 10) + 1
        for i, rec in enumerate(scaler_recs):
            # json-safe conversion of config data
            # non-json elements are converted via SymbolSpec
            rec_cfg = self._json_safe_cfg(config=rec.get_config(), ctx=ctx)

            # optional artifact reference (for tracking only)
            if rec.scaler_obj is not None:
                art_path = rec.scaler_obj.save(
                    filepath=save_dir / "scalers" / f"scaler_{i:03d}",
                    overwrite=False,
                )
                rec_cfg["_scaler_artifact"] = str(
                    Path(art_path).relative_to(save_dir),
                )

            cfg_path = dir_scaler_rec / f"scaler_{i:0{n_digits}d}.json"
            self._write_json(rec_cfg, cfg_path)
            scaler_cfg_files.append(cfg_path.name)

        file_mapping["scaler_records"] = scaler_cfg_files

        # ------------------------------------------------
        # Save SplitterRecords
        #  - only configs used during reload (resplit)
        #  - full artifact saved for tracking only
        # ------------------------------------------------
        dir_splitter_rec = Path(save_dir) / "splitter_records"
        dir_splitter_rec.mkdir(exist_ok=True)

        splitter_cfg_files: list[str] = []
        splitter_recs: list[SplitterRecord] = state.pop("splitter_records")
        n_digits = (len(splitter_recs) // 10) + 1
        for i, rec in enumerate(splitter_recs):
            # json-safe conversion of config data
            # non-json elements are converted via SymbolSpec
            rec_cfg = self._json_safe_cfg(config=rec.get_config(), ctx=ctx)

            # optional artifact reference (for tracking only)
            if rec.splitter is not None:
                art_path = rec.splitter.save(
                    filepath=save_dir / "splitters" / f"splitter_{i:0{n_digits}d}",
                    overwrite=False,
                )
                rec_cfg["_splitter_artifact"] = str(
                    Path(art_path).relative_to(save_dir),
                )

            cfg_path = dir_splitter_rec / f"splitter_{i:0{n_digits}d}.json"
            self._write_json(rec_cfg, cfg_path)
            splitter_cfg_files.append(cfg_path.name)

        file_mapping["splitter_records"] = splitter_cfg_files

        # ------------------------------------------------
        # Other state values
        #  - extra arguments (eg, "super") will be handled
        #    via the default state.pkl
        # ------------------------------------------------
        if state:
            # Save to pickle
            file_state = Path(save_dir) / self.state_rel_path
            with Path.open(file_state, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            file_mapping["state"] = self.state_rel_path

        return file_mapping

    # ================================================
    # Splitter Decoding
    # ================================================
    def decode(
        self,
        cls: type[FeatureSet],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> FeatureSet:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[FeatureSet]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            FeatureSet: The re-instantiated object.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        # ------------------------------------------------
        # Decode Config
        # ------------------------------------------------
        config: dict[str, Any] = self.decode_config(load_dir=load_dir, ctx=ctx)

        # ------------------------------------------------
        # Decode State
        # ------------------------------------------------
        state: dict[str, Any] = self.decode_state(load_dir=load_dir, ctx=ctx)

        # ------------------------------------------------
        # Instantiate FeatureSet
        #  - Cannot register until performing collision checks
        #  - Reconstruct splits and scalers from config
        #  - Scalers get re-fit and re-applied in order
        # ------------------------------------------------
        fs_obj = cls.from_config(config=config, register=False)

        # Restore splits (provide source)
        split_cfgs: dict[str, Any] = state.pop("split_configs")
        splits: dict[str, FeatureSetView] = {
            k: FeatureSetView(
                source=fs_obj,
                indices=cfg["indices"],
                columns=cfg["columns"],
                label=cfg["label"],
            )
            for k, cfg in split_cfgs.items()
        }
        state["splits"] = splits

        # Set state
        fs_obj.set_state(state=state)

        # ------------------------------------------------
        # Collision Checking
        # ------------------------------------------------
        fs_obj = self.handle_node_collision(obj=fs_obj, ctx=ctx)

        # ------------------------------------------------
        # Re-fit Scalers
        # - While the transformed data and splits are identical after set_state,
        #   some scaler states can't be fully serialized. This means that reusing the
        #   de-serialized scaler (e.g., using `undo_transform`) could cause instability.
        # - To avoid this, we re-apply the scalers in the same order (ie re-fit them)
        # ------------------------------------------------
        # 1. Remove any residual transformed representations (we fully rebuild from raw rep)
        for domain in [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS]:
            for k in fs_obj.collection._get_domain_keys(
                domain=domain,
                include_domain_prefix=False,
                include_rep_suffix=False,
            ):
                if REP_TRANSFORMED in fs_obj.collection._get_rep_keys(
                    domain=domain,
                    key=k,
                ):
                    fs_obj.collection.delete_rep(
                        domain=domain,
                        key=k,
                        rep=REP_TRANSFORMED,
                    )

        # 2. Reapply scalers (clear those on featureset)
        fs_obj._scaler_recs = []
        records: list[ScalerRecord] = state["scaler_records"]
        for rec in records:
            fs_obj.fit_transform(
                scaler=rec.scaler_obj,
                domain=rec.domain,
                keys=rec.keys,
                fit_to_split=rec.fit_split,
            )

        return fs_obj

    def decode_state(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            dict[str, Any]: The decoded state data.

        """
        from modularml.core.data.sample_collection import SampleCollection
        from modularml.core.splitting.splitter_record import SplitterRecord
        from modularml.core.transforms.scaler_record import ScalerRecord

        # Extract file mapping
        load_dir = Path(load_dir)
        file_mapping: dict[str, Any] = self._read_json(
            load_dir / "artifact.json",
        )["files"]

        # ------------------------------------------------
        # Restore basic state props
        # ------------------------------------------------
        state: dict[str, Any] = {}
        if "state" in file_mapping:
            file_state: Path = load_dir / file_mapping["state"]
            with Path.open(file_state, "rb") as f:
                state: dict[str, Any] = pickle.load(f)

        # ------------------------------------------------
        # Restore SampleCollection
        # ------------------------------------------------
        file_coll: Path = load_dir / file_mapping["sample_collection"]
        coll: SampleCollection = SampleCollection.load(file_coll)
        state["sample_collection"] = coll

        # Add hash for later validation
        state["table_hash"] = hash_pyarrow_table(coll.table)

        # ------------------------------------------------
        # Restore Split Configs
        # ------------------------------------------------
        file_split_cfgs: Path = load_dir / file_mapping["split_configs"]
        state["split_configs"] = self._read_json(file_split_cfgs)

        # ------------------------------------------------
        # Restore SplitterRecords
        # ------------------------------------------------
        files_split_recs: list[Path] = [
            (load_dir / "splitter_records" / x)
            for x in file_mapping["splitter_records"]
        ]
        split_recs: list[SplitterRecord] = []
        for cfg_path in files_split_recs:
            raw_cfg = self._read_json(cfg_path)

            # Restore any SymbolSpecs
            rec_cfg = self._restore_json_cfg(data=raw_cfg, ctx=ctx)

            # Remove reference to stored artifact (not used in reloading)
            _ = rec_cfg.pop("_splitter_artifact", None)

            # Construct SplitterRecord
            split_recs.append(SplitterRecord.from_config(rec_cfg))

        state["splitter_records"] = split_recs

        # ------------------------------------------------
        # Restore ScalerRecords
        # ------------------------------------------------
        files_scaler_recs: list[Path] = [
            (load_dir / "scaler_records" / x) for x in file_mapping["scaler_records"]
        ]
        scaler_recs: list[SplitterRecord] = []
        for cfg_path in files_scaler_recs:
            raw_cfg = self._read_json(cfg_path)

            # Restore any SymbolSpecs
            rec_cfg = self._restore_json_cfg(data=raw_cfg, ctx=ctx)

            # Remove reference to stored artifact (not used in reloading)
            _ = rec_cfg.pop("_scaler_artifact", None)

            # Construct ScalerRecord
            scaler_recs.append(ScalerRecord.from_config(rec_cfg))

        state["scaler_records"] = scaler_recs

        return state
