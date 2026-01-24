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
        key = (object_type, from_version)

        if key in self._migrations and not overwrite:
            msg = f"Migration already registered for {object_type} {from_version} -> {self._migrations[key][0]}"
            raise ValueError(msg)

        self._migrations[key] = (to_version, fn)

    # ================================================
    # Introspection
    # ================================================
    def has_migration(self, object_type: str, version: str) -> bool:
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
        Apply all registered migrations to an Artifact until up-to-date.

        Returns:
            New Artifact with migrated config/state/version.

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


"""
Example migration function to migrate FeatureSet version 1.0 to 1.1


We define a function that maps the old artifact to the new artifact
```
def migrate_featureset_1_0_to_1_1(
    *,
    artifact: Artifact,
    artifact_path: Path,
    handler,
) -> Artifact:
    # Load config
    config_path = artifact_path / artifact.files["config"]
    handle
    config = handler.decode_config(config_dir=artifact_path)

    # Perform migration
    if "scalers" in config:
        config["transforms"] = config.pop("scalers")

    # Write back
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    return artifact
```

Then we simply register it:
```
migration_registry.register(
    object_type="FeatureSet",
    from_version="1.0",
    to_version="1.1",
    fn=migrate_featureset_1_0_to_1_1,
)
```
"""
