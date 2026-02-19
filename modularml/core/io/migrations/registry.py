"""Registry for applying artifact migrations."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from modularml.core.io.artifacts import Artifact
from modularml.core.io.handlers.handler import BaseHandler

# A migration transforms (artifact, artifact_path, handler) -> Artifact
MigrationFnc = Callable[[Artifact, Path, BaseHandler], Artifact]


class MigrationRegistry:
    """
    Registry of object-level migrations.

    Migrations are applied to convert one artifact to another.

    Attributes:
        _migrations (dict[tuple[str, str], tuple[str, MigrationFnc]]):
            Mapping from `(object_type, from_version)` to `(to_version, migration_fn)`.

    """

    def __init__(self) -> None:
        # (object_type, from_version) -> (to_version, fn)
        self._migrations: dict[
            tuple[str, str],
            tuple[str, MigrationFnc],
        ] = {}

    # ============================================
    # Registration
    # ============================================
    def register(
        self,
        *,
        object_type: str,
        from_version: str,
        to_version: str,
        fn: MigrationFnc,
        overwrite: bool = False,
    ) -> None:
        """
        Register a migration function for a specific object type/version.

        Args:
            object_type (str): Serialized kind identifier (for example, `fs`).
            from_version (str): Source version handled by the migration.
            to_version (str): Target version after migration.
            fn (MigrationFnc): Callable that mutates the artifact in-place.
            overwrite (bool): Replace existing registration if True.

        Returns:
            None

        Raises:
            ValueError: If a migration already exists and `overwrite` is False.

        """
        key = (object_type, from_version)

        if key in self._migrations and not overwrite:
            msg = f"Migration already registered for {object_type} {from_version} -> {self._migrations[key][0]}"
            raise ValueError(msg)

        self._migrations[key] = (to_version, fn)

    # ================================================
    # Introspection
    # ================================================
    def has_migration(self, object_type: str, version: str) -> bool:
        """
        Check whether a migration exists for an object/version pair.

        Args:
            object_type (str): Artifact kind identifier.
            version (str): Current object version.

        Returns:
            bool: True if a migration is registered.

        """
        return (object_type, version) in self._migrations

    # ================================================
    # Execution
    # ================================================
    def apply(
        self,
        artifact: Artifact,
        *,
        artifact_path: Path,
        handler: BaseHandler,
    ) -> Artifact:
        """
        Apply registered migrations to `artifact` until the version is current.

        Args:
            artifact (Artifact): Artifact to migrate.
            artifact_path (Path): Directory containing the artifact files.
            handler (BaseHandler): Handler responsible for encoding/decoding.

        Returns:
            Artifact: Migrated artifact with updated header metadata.

        """
        header = artifact.header
        object_type = header.kind
        version = header.object_version

        while (object_type, version) in self._migrations:
            next_version, fn = self._migrations[(object_type, version)]

            artifact = fn(
                artifact=artifact,
                artifact_path=artifact_path,
                handler=handler,
            )

            header = replace(
                artifact.header,
                object_version=next_version,
            )
            artifact = replace(artifact, header=header)
            version = next_version

        return artifact


migration_registry = MigrationRegistry()
