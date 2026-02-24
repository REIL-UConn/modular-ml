"""High-level serializer for saving/loading ModularML artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from modularml.core.io.artifacts import Artifact, ArtifactHeader
from modularml.core.io.conventions import KindRegistry, kind_registry
from modularml.core.io.handlers.registry import handler_registry
from modularml.core.io.migrations.registry import migration_registry
from modularml.core.io.packaged_code_loaders.default_loader import (
    default_packaged_code_loader,
)
from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy
from modularml.core.io.symbol_registry import (
    SymbolRegistry,
    SymbolResolutionError,
    symbol_registry,
)
from modularml.core.io.symbol_spec import SymbolSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from modularml.core.io.handlers.handler import HandlerRegistry


def _zip_dir(src: Path, dest: Path) -> None:
    """
    Archive `src` directory into a ZIP file at `dest`.

    Args:
        src (Path): Directory to compress.
        dest (Path): Output ZIP file path.

    """
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))


def _unzip(src: Path, dest: Path) -> None:
    """
    Extract ZIP file `src` into directory `dest`.

    Args:
        src (Path): Source ZIP file.
        dest (Path): Destination directory for extracted contents.

    """
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest)


def _enforce_file_suffix(path: Path, cls: type) -> Path:
    """
    Ensure artifacts follow the `<kind>.mml` suffix for `cls`.

    Args:
        path (Path): Requested output path (suffix may change).
        cls (type): Class whose serialization kind determines the suffix.

    Returns:
        Path: Path with the enforced suffix; parent directories will exist.

    """
    # Enforce 'kind.mml' suffix
    path = Path(path)
    exp_suffix = kind_registry.get_kind(cls=cls).file_suffix
    if "".join(path.suffixes) != exp_suffix:
        path = path.with_name(path.stem.split(".")[0] + exp_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


class SaveContext:
    """
    Helper context passed to handlers during serialization.

    Attributes:
        artifact_path (Path): Root folder of the artifact being written.
        serializer (Serializer): Owning serializer instance.
        mml_version (str): Version string embedded in emitted artifacts.

    """

    def __init__(
        self,
        artifact_path: Path,
        serializer: Serializer,
        mml_version: str = "1.0.0",
    ):
        """
        Initialize a :class:`SaveContext`.

        Args:
            artifact_path (Path): Root folder where artifacts are written.
            serializer (Serializer): Owning serializer instance.
            mml_version (str): Version string recorded in emitted manifests.

        """
        self.artifact_path = artifact_path
        self.serializer = serializer
        self.mml_version = mml_version

        self._packaged_symbols: dict[object, SymbolSpec] = {}

    # =================================================
    # Symbol packaging and spec creation
    # =================================================
    def _package_symbol(self, symbol: object) -> SymbolSpec:
        """
        Package `symbol` into the artifact and return its :class:`SymbolSpec`.

        Args:
            symbol (object): Class or function requiring packaging.

        Returns:
            SymbolSpec: Packaged symbol specification.

        Raises:
            RuntimeError: If the symbol cannot be packaged (lambda, closures, or __main__).
            FileNotFoundError: If the source file is missing.
            TypeError: If the symbol type is unsupported.

        """
        # Resolve instances to its class (functions should be unchanged)
        if not (inspect.isclass(symbol) or inspect.isfunction(symbol)):
            symbol = symbol.__class__

        # Normalize symbol identity
        if not hasattr(symbol, "__qualname__") or not hasattr(symbol, "__module__"):
            msg = f"Unsupported symbol type: {type(symbol)}"
            raise TypeError(msg)

        qualname = symbol.__qualname__
        module = symbol.__module__

        if symbol in self._packaged_symbols:
            return self._packaged_symbols[symbol]

        # Reject unsupported symbols
        if inspect.isfunction(symbol):
            if symbol.__name__ == "<lambda>":
                msg = (
                    "Cannot package lambda functions. Define a named function "
                    "in a .py file."
                )
                raise RuntimeError(msg)

            if inspect.getclosurevars(symbol).nonlocals:
                msg = f"Cannot package function '{qualname}' with closures."
                raise RuntimeError(msg)
        if module == "__main__":
            msg = (
                f"Cannot package symbol '{qualname}' defined in '__main__'. "
                "Define it in a standalone Python file."
            )
            raise RuntimeError(msg)

        # Prepare destination
        code_dir = self.artifact_path / "code"
        code_dir.mkdir(exist_ok=True)

        # Helpers: resolve destination filename safely
        def _hash_text(text: str) -> str:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        def _resolve_dest_file(base_name: str, source_txt: str) -> Path:
            base = Path(base_name)
            candidate = code_dir / base.name

            # Case 1: filename unused
            if not candidate.exists():
                return candidate

            # Case 2: filename exists & contain same content
            existing_txt = candidate.read_text(encoding="utf-8")
            if _hash_text(existing_txt) == _hash_text(source_txt):
                return candidate

            # Case 3: conflicting filename -> append suffix
            i = 0
            while True:
                candidate = code_dir / f"{base.stem}_{i}{base.suffix}"
                if not candidate.exists():
                    return candidate
                i += 1

        # Case 1: symbol came from packaged loader
        if hasattr(
            symbol,
            "__mml_source_text__",
        ) and hasattr(
            symbol,
            "__mml_source_ref__",
        ):
            source_text: str = symbol.__mml_source_text__
            source_ref: str = symbol.__mml_source_ref__

            rel_path, _ = source_ref.split(":", 1)
            base_name = Path(rel_path).name
        # Case 2: normal symbol -> inspect source file
        else:
            try:
                src_file = Path(inspect.getfile(symbol))
            except (TypeError, OSError) as exc:
                msg = (
                    f"Cannot bundle source for symbol '{qualname}'. Source "
                    "file is not resolvable."
                )
                raise RuntimeError(msg) from exc
            if not src_file.exists():
                msg = f"Source file not found: {src_file}"
                raise FileNotFoundError(msg)
            source_text = src_file.read_text(encoding="utf-8")
            base_name = src_file.name

        # Get destination filepath & write if not already written
        dest_file = _resolve_dest_file(
            base_name=base_name,
            source_txt=source_text,
        )
        if not dest_file.exists():
            dest_file.write_text(source_text, encoding="utf-8")

        # Build SymbolSpec
        source_ref = f"code/{dest_file.name}:{qualname}"
        spec = SymbolSpec(
            policy=SerializationPolicy.PACKAGED,
            module=None,  # packaged code is not imported by module path
            qualname=qualname,
            source_ref=source_ref,
        )
        spec.validate()

        self._packaged_symbols[symbol] = spec
        return spec

    def make_symbol_spec(
        self,
        symbol: object,
        *,
        policy: SerializationPolicy,
    ) -> SymbolSpec:
        """
        Construct a :class:`SymbolSpec` describing `symbol`.

        Args:
            symbol (object): Class or function to describe.
            policy (SerializationPolicy): Desired serialization policy.

        Returns:
            SymbolSpec: Specification compatible with the selected policy.

        Raises:
            TypeError: If the policy is unsupported.

        """
        policy = normalize_policy(policy)

        if policy is SerializationPolicy.PACKAGED:
            spec = self._package_symbol(symbol=symbol)

        elif policy is SerializationPolicy.BUILTIN:
            spec = SymbolSpec(
                policy=SerializationPolicy.BUILTIN,
                key=symbol_registry.key_for(symbol),
            )

        elif policy is SerializationPolicy.REGISTERED:
            r_path, r_key = symbol_registry.registered_location_for(symbol)
            spec = SymbolSpec(
                policy=SerializationPolicy.REGISTERED,
                registry_path=r_path,
                registry_key=r_key,
            )

        elif policy is SerializationPolicy.STATE_ONLY:
            spec = SymbolSpec(policy=SerializationPolicy.STATE_ONLY)

        else:
            msg = f"Unsupported policy: {policy}"
            raise TypeError(msg)

        spec.validate()
        return spec

    # =================================================
    # Artifact writing
    # =================================================
    def write_artifact(
        self,
        obj: Any,
        symbol_spec: SymbolSpec,
        save_dir: Path,
    ):
        """
        Write a serialized artifact for `obj` into `save_dir`.

        Args:
            obj (Any): Object to serialize.
            symbol_spec (SymbolSpec): Specification returned by :meth:`make_symbol_spec`.
            save_dir (Path): Directory receiving the encoded files.

        """
        # Get handler for this object type
        handler = handler_registry.resolve(type(obj))

        # Encode config and state files with handler
        file_mapping = handler.encode(obj=obj, save_dir=save_dir, ctx=self)

        # Create artifact for root
        artifact = Artifact(
            header=ArtifactHeader(
                mml_version=self.mml_version,
                schema_version=Artifact.schema_version,
                object_version=handler.object_version,
                kind=kind_registry.get_kind(type(obj)).kind,
                symbol_spec=symbol_spec.to_dict(),
            ),
            files=file_mapping,
        )

        # Write artifact to disk (as json)
        with (save_dir / "artifact.json").open("w", encoding="utf-8") as f:
            json.dump(artifact.to_json(), f, indent=2, sort_keys=True)

    def emit_mml(
        self,
        obj: Any,
        symbol_spec: SymbolSpec,
        out_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Write a zipped artifact to `out_path`.

        Args:
            obj (Any): Object to serialize.
            symbol_spec (SymbolSpec): Specification returned by :meth:`make_symbol_spec`.
            out_path (Path): Target file path (suffix may be adjusted).
            overwrite (bool): Overwrite `out_path` when True.

        Returns:
            Path: Path to the written `.mml` artifact.

        Raises:
            FileExistsError: If `out_path` exists and `overwrite` is False.

        """
        # Enforce 'kind.mml' suffix
        out_path = _enforce_file_suffix(out_path, cls=type(obj))
        if out_path.exists() and not overwrite:
            msg = f"Artifact already exists at: {out_path}"
            raise FileExistsError(msg)

        # Write encodings and artifact in temp directory, then compress to zip
        with TemporaryDirectory() as subtmp:
            subdir = Path(subtmp)

            # Write artifact for this obj
            # This internally calls handler.encode on subdir
            self.write_artifact(
                obj=obj,
                symbol_spec=symbol_spec,
                save_dir=subdir,
            )

            # Compress subdir to a zip file and save to out_path
            _zip_dir(subdir, out_path)

        return out_path


class LoadContext:
    """
    Context provided to BaseHandlers during deserialization.

    Attributes:
        artifact_path (Path):
            Root directory of the artifact being loaded.
        serializer (Any):
            Owning serializer instance (for advanced resolution if needed).

        allow_packaged_code (bool):
            Whether executing bundled code is permitted.
        overwrite_collision (bool):
            Whether to overwrite collision (ie. same `node_id`) or generate a new
            node_id. Defaults to False (ie. register with new node_id)

    """

    def __init__(
        self,
        artifact_path: Path,
        serializer: Serializer,
        *,
        allow_packaged_code: bool,
        overwrite_collision: bool = False,
        extras: dict[str, Any] | None = None,
    ):
        """
        Initialize a :class:`LoadContext`.

        Args:
            artifact_path (Path): Root directory of the artifact being loaded.
            serializer (Serializer): Owning serializer instance.
            allow_packaged_code (bool): Whether executing packaged code is permitted.
            overwrite_collision (bool): Whether to overwrite ID collisions on load.
            extras (dict[str, Any] | None): Optional user-provided metadata.

        """
        self.artifact_path = artifact_path
        self.serializer = serializer
        self.allow_packaged_code = allow_packaged_code
        self.overwrite_collision = overwrite_collision
        self.file_mapping: dict[str, str | None] | None = None
        self.extras: dict[str, Any] = extras or {}

    def load_from_dir(
        self,
        dir_load: Path,
        packaged_code_loader: Callable[[str], object] | None = None,
        provided_cls: type | None = None,
    ):
        """
        Load an artifact from a directory previously produced by :meth:`write_artifact`.

        Args:
            dir_load (Path): Directory containing the artifact files.
            packaged_code_loader (Callable[[str], object] | None): Loader for packaged code.
            provided_cls (type | None): Required when loading :attr:`SerializationPolicy.STATE_ONLY` artifacts.

        Returns:
            Any: Reconstructed object.

        Raises:
            FileNotFoundError: If `mml_file` does not exist.

        Raises:
            FileNotFoundError: If `artifact.json` is missing.
            SymbolResolutionError: If the stored symbol cannot be resolved.

        """
        if packaged_code_loader is None:

            def packaged_code_loader(source_ref):
                return default_packaged_code_loader(
                    artifact_path=dir_load,
                    source_ref=source_ref,
                    allow_packaged=self.allow_packaged_code,
                )

        # Validate artifact existence
        artifact_json = dir_load / "artifact.json"
        if not artifact_json.exists():
            raise FileNotFoundError("artifact.json missing from archive")

        # Read artifact
        with artifact_json.open("r", encoding="utf-8") as f:
            artifact = Artifact.from_json(json.load(f))

        # Extract SymbolSpec
        spec = SymbolSpec.from_dict(artifact.header.symbol_spec)
        spec.validate()

        # Resolve class (no object instantiation yet)
        if spec.policy is SerializationPolicy.STATE_ONLY:
            if provided_cls is None:
                raise SymbolResolutionError(
                    "STATE_ONLY artifacts require `provided_cls`.",
                )
            cls = provided_cls
        else:
            cls = symbol_registry.resolve_symbol(
                spec,
                allow_packaged_code=self.allow_packaged_code,
                packaged_code_loader=packaged_code_loader,
            )

        # Apply any registered migrations
        artifact = migration_registry.apply(
            artifact=artifact,
            artifact_path=dir_load,
            handler=handler_registry.resolve(cls=cls),
        )
        self.file_mapping = artifact.files

        # Reconstruct object (handler class handles this logic)
        handler = handler_registry.resolve(cls)
        obj = handler.decode(cls=cls, load_dir=dir_load, ctx=self)

        return obj

    def load_from_mml(
        self,
        mml_file: Path,
        packaged_code_loader: Callable[[str], object] | None = None,
        provided_cls: type | None = None,
    ):
        """
        Load an artifact directly from a zipped `.mml` archive.

        Args:
            mml_file (Path): Path to the `.mml` archive.
            packaged_code_loader (Callable[[str], object] | None): Loader for packaged code.
            provided_cls (type | None): Required when loading :attr:`SerializationPolicy.STATE_ONLY` artifacts.

        Returns:
            Any: Reconstructed object.

        """
        path: Path = Path(mml_file)
        if not path.exists():
            raise FileNotFoundError(path)

        if packaged_code_loader is None:

            def packaged_code_loader(source_ref):
                return default_packaged_code_loader(
                    artifact_path=mml_file,
                    source_ref=source_ref,
                    allow_packaged=self.allow_packaged_code,
                )

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _unzip(path, tmp_path)

            return self.load_from_dir(
                dir_load=tmp_path,
                packaged_code_loader=packaged_code_loader,
                provided_cls=provided_cls,
            )


class Serializer:
    """
    Central serializer that saves and loads ModularML objects.

    The serializer:
    - determines artifact kind (KindRegistry)
    - determines class identity (SymbolRegistry + SerializationPolicy)
    - encodes config/state (BaseHandlers, Configurable, Stateful)
    - writes to disk and reconstructs on load

    Args:
        kind_registry (KindRegistry):
            Registry for file suffix/kind mapping.
        symbol_registry (SymbolRegistry):
            Registry for SymbolSpec to/from class resolution.
        handler_registry (HandlerRegistry):
            Registry for BaseHandlers.
        mml_version (str):
            ModularML version string to embed in artifacts.
        schema_version (int):
            Artifact schema version for migrations.

    """

    def __init__(
        self,
        *,
        kind_registry: KindRegistry,
        symbol_registry: SymbolRegistry,
        handler_registry: HandlerRegistry,
        mml_version: str = "0.0.0",
    ):
        self.kind_registry = kind_registry
        self.symbol_registry = symbol_registry
        self.handler_registry = handler_registry
        self.mml_version = mml_version

    # ================================================
    # Save
    # ================================================
    def save(
        self,
        obj: Any,
        save_path: str,
        *,
        policy: SerializationPolicy,
        overwrite: bool = False,
    ) -> str:
        """
        Serialize `obj` to disk as a ModularML artifact.

        Args:
            obj (Any): Object to serialize.
            save_path (str): Output path for the resulting `.mml` file.
            policy (SerializationPolicy): Class resolution policy for `obj`.
            overwrite (bool): Overwrite existing artifacts when True.

        Returns:
            str: Path to the written artifact file.

        Raises:
            FileExistsError: If `save_path` exists and `overwrite` is False.

        """
        # Enforce 'kind.mml' suffix
        save_path = _enforce_file_suffix(save_path, cls=type(obj))
        if save_path.exists() and not overwrite:
            msg = f"Artifact already exists at: {save_path}"
            raise FileExistsError(msg)

        # Wrap everything in temp directory
        # This gets cast into a zip file at the end
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create SaveContext for any recursive calls
            ctx = SaveContext(
                artifact_path=tmp_path,
                serializer=self,
                mml_version=self.mml_version,
            )

            # Write artifact
            symbol_spec = ctx.make_symbol_spec(
                symbol=type(obj),
                policy=policy,
            )
            ctx.write_artifact(
                obj=obj,
                symbol_spec=symbol_spec,
                save_dir=tmp_path,
            )

            # Compress tmp_dir to a zip file and save to save_path
            _zip_dir(tmp_path, save_path)

        return str(save_path)

    # ================================================
    # Load
    # ================================================
    def load(
        self,
        path: str,
        *,
        allow_packaged_code: bool = False,
        provided_class: type | None = None,
        overwrite: bool = False,
        extras: dict[str, Any] | None = None,
    ) -> Any:
        """
        Load an artifact from disk and reconstruct the serialized object.

        Args:
            path (str): Path to the `.mml` artifact.
            allow_packaged_code (bool): Whether executing packaged source code is allowed.
            provided_class (type | None): Required when loading :attr:`SerializationPolicy.STATE_ONLY` artifacts.
            overwrite (bool): Overwrite conflicting node registrations in the active context.
            extras (dict[str, Any] | None): Additional metadata forwarded via :class:`LoadContext`.

        Returns:
            Any: Reconstructed object.

        Raises:
            FileNotFoundError: If `path` does not exist.

        """
        path: Path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _unzip(path, tmp_path)

            # Create LoadContext for any recursive calls
            ctx = LoadContext(
                artifact_path=tmp_path,
                serializer=self,
                allow_packaged_code=allow_packaged_code,
                overwrite_collision=overwrite,
                extras=extras,
            )

            return ctx.load_from_dir(
                dir_load=tmp_path,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=tmp_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
                provided_cls=provided_class,
            )


serializer = Serializer(
    kind_registry=kind_registry,
    symbol_registry=symbol_registry,
    handler_registry=handler_registry,
    mml_version="1.0.0",
)
