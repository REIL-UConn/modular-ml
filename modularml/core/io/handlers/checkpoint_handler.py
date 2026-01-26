from __future__ import annotations

import logging
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
    def encode_state(
        self,
        obj: Checkpoint,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes Checkpoint state.

        Args:
            obj (Checkpoint):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        data = {}
        for key, entry in obj.entries.items():
            handler = ctx.serializer.handler_registry.resolve(entry.entry_cls)

            # Each entry separately encodes its state
            # The file where the entry state is saved is returned
            subdir = Path(save_dir) / key.replace(":", "~")
            subdir.mkdir(parents=True, exist_ok=True)
            _ = handler.encode_state(
                obj=entry.entry_obj,
                save_dir=subdir,
                ctx=ctx,
            )

            # Stores the class and path to saved state of this entry
            data[key] = {
                "class": handler.get_symbol_spec(
                    entry.entry_obj,
                    ctx=ctx,
                ).to_dict(),
                "state_dir": str(subdir.relative_to(save_dir)),
            }

        # Write all checkpointed path to a json config
        save_path = self._write_json(
            data=data,
            save_path=Path(save_dir) / self.state_rel_path,
        )
        return {"state": str(save_path.relative_to(save_dir))}

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
        # Reload entries from state
        state = self.decode_state(
            load_dir=load_dir,
            ctx=ctx,
        )
        entries: dict[str, CheckpointEntry] = {}
        for k, v in state.items():
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

            # Decode the saved state for this entry
            k_state_path = Path(load_dir) / v["state_dir"]
            handler = ctx.serializer.handler_registry.resolve(k_cls)
            k_state = handler.decode_state(load_dir=k_state_path, ctx=ctx)

            entries[k] = CheckpointEntry(
                entry_cls=k_cls,
                entry_state=k_state,
                entry_obj=None,  # can't restore object since never saved config
            )

        # Create Checkpoint instance
        return cls(entries=entries)

    def decode_state(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,  # noqa: ARG002
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
        ckpt_data = self._read_json(Path(load_dir) / self.state_rel_path)
        return ckpt_data
