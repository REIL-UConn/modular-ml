from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from modularml.core.io.checkpoint import Checkpoint, CheckpointEntry
from modularml.core.io.handlers.handler import BaseHandler
from modularml.core.io.packaged_code_loaders.default_loader import (
    default_packaged_code_loader,
)
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.symbol_spec import SymbolSpec
from modularml.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext

T = TypeVar("T")

logger = get_logger(level=logging.INFO)


class CheckpointHandler(BaseHandler[Checkpoint]):
    """Handles serialization of Checkpoint objects."""

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "checkpoint.json"

    # ================================================
    # Encoding
    # ================================================
    def encode(
        self,
        obj: Checkpoint,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes Checkpoint data.

        Args:
            obj (Checkpoint):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: File mapping of encoded data.

        """
        save_dir = Path(save_dir)
        file_mapping: dict[str, Any] = {}

        # Encode meta data (use pickle)
        dir_meta = save_dir / "meta"
        dir_meta.mkdir(exist_ok=True)
        file_meta = dir_meta / "ckpt_meta.pkl"
        with Path.open(file_meta, "wb") as f:
            pickle.dump(obj.meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        file_mapping["meta"] = str(file_meta.relative_to(save_dir))

        # Encode entries (stateful objects)
        # Each entry has its state written using its own handler (if defined)
        # Then we serialize the entry class as well
        dir_entries = save_dir / "entries"
        dir_entries.mkdir(exist_ok=True)

        entry_cls_data = {}
        for key, entry in obj.entries.items():
            handler = ctx.serializer.handler_registry.resolve(entry.entry_cls)

            # Each entry separately encodes its state
            # The file where the entry state is saved is returned
            dir_k = dir_entries / key.replace(":", "~")
            dir_k.mkdir(parents=True, exist_ok=True)
            _ = handler.encode_state(
                obj=entry.entry_obj,
                save_dir=dir_k,
                ctx=ctx,
            )

            # Stores the class and path to saved state of this entry
            entry_cls_data[key] = {
                "class": handler.get_symbol_spec(
                    entry.entry_obj,
                    ctx=ctx,
                ).to_dict(),
                "state_dir": str(dir_k.relative_to(save_dir)),
            }
        save_path = self._write_json(
            data=entry_cls_data,
            save_path=dir_entries / "entry_class_config.json",
        )
        file_mapping["entry_cls_data"] = str(save_path.relative_to(save_dir))

        return file_mapping

    # ================================================
    # Decoding
    # ================================================
    def decode(
        self,
        cls: Checkpoint,
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> Checkpoint:
        """
        Decodes a Checkpoint from a saved artifact.

        Args:
            cls (Checkpoint):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            Checkpoint: The re-instantiated checkpoint.

        """
        load_dir = Path(load_dir)

        # Read file mapping from artifact.json
        artifact_data = self._read_json(load_dir / "artifact.json")
        file_mapping: dict[str, Any] = artifact_data["files"]

        # Reload meta data
        file_meta = load_dir / file_mapping["meta"]
        if not file_meta.exists():
            msg = f"Could not find meta-data file in directory: '{file_meta}'."
            raise FileNotFoundError(msg)
        with Path.open(file_meta, "rb") as f:
            meta: dict[str, Any] = pickle.load(f)

        # Reload entries
        # 1. Reload class data
        file_cls_data = load_dir / file_mapping["entry_cls_data"]
        if not file_cls_data.exists():
            msg = (
                f"Could not find entry class data file in directory: '{file_cls_data}'."
            )
            raise FileNotFoundError(msg)
        entry_cls_data = self._read_json(file_cls_data)

        # 2. Reload entries
        entries: dict[str, CheckpointEntry] = {}
        for k, v in entry_cls_data.items():
            # Resolve class for this entry
            k_cls = symbol_registry.resolve_symbol(
                spec=SymbolSpec.from_dict(v["class"]),
                allow_packaged_code=ctx.allow_packaged_code,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=ctx.artifact_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
            )

            # Decode saved state
            k_state_path = load_dir / v["state_dir"]
            handler = ctx.serializer.handler_registry.resolve(k_cls)
            k_state = handler.decode_state(load_dir=k_state_path, ctx=ctx)

            entries[k] = CheckpointEntry(
                entry_cls=k_cls,
                entry_state=k_state,
                entry_obj=None,  # can't restore object since never saved config
            )

        # Create checkpoint instance
        return cls(
            entries=entries,
            meta=meta,
        )
