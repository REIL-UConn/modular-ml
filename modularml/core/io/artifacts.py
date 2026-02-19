"""Artifact manifest definitions for ModularML serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ArtifactHeader:
    """
    Metadata describing a saved ModularML artifact for robust round-trip loading.

    Attributes:
        mml_version (str): ModularML version used to save the artifact.
        schema_version (str): Artifact schema version.
        object_version (str): Version of the serialized object.
        kind (str): Kind code (for example, `fs` or `mg`) used for naming conventions.
        symbol_spec (dict[str, Any]): Serialized :class:`SymbolSpec` payload.

    """

    mml_version: str
    schema_version: str
    object_version: str
    kind: str
    symbol_spec: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """
        Serialize the header into a JSON-compatible dictionary.

        Returns:
            dict[str, Any]: JSON-friendly representation of this header.

        """
        return {
            "mml_version": self.mml_version,
            "schema_version": self.schema_version,
            "object_version": self.object_version,
            "kind": self.kind,
            "symbol_spec": self.symbol_spec,
        }

    @classmethod
    def from_json(cls, json: dict[str, Any]) -> ArtifactHeader:
        """
        Deserialize a header from JSON data.

        Args:
            json (dict[str, Any]): Serialized header data.

        Returns:
            ArtifactHeader: Reconstructed artifact header.

        """
        return cls(
            mml_version=json["mml_version"],
            schema_version=json["schema_version"],
            object_version=json["object_version"],
            kind=json["kind"],
            symbol_spec=json["symbol_spec"],
        )


@dataclass(frozen=True)
class Artifact:
    """
    Artifact manifest describing where all serialized components live.

    Attributes:
        header (ArtifactHeader): Artifact metadata.
        files (dict[str, Any]): Logical file map (e.g., `{'config': 'config.json'}`).
        schema_version (str): Artifact version string (defaults to `1.0`).

    """

    header: ArtifactHeader
    files: dict[str, Any]
    schema_version: str = "1.0"

    def to_json(self) -> dict[str, Any]:
        """
        Serialize the artifact manifest into JSON-compatible data.

        Returns:
            dict[str, Any]: JSON-friendly representation of the manifest.

        """
        return {
            "header": self.header.to_json(),
            "files": self.files,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_json(cls, json: dict[str, Any]) -> Artifact:
        """
        Deserialize an artifact manifest from JSON data.

        Args:
            json (dict[str, Any]): Serialized manifest data.

        Returns:
            Artifact: Reconstructed manifest instance.

        """
        return cls(
            header=ArtifactHeader.from_json(json["header"]),
            files=json["files"],
            schema_version=json["schema_version"],
        )
