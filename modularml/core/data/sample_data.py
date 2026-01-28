from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_OUTPUTS,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.utils.data.conversion import convert_to_format
from modularml.utils.data.data_format import (
    _TENSORLIKE_FORMATS,
    DataFormat,
    format_is_tensorlike,
    get_data_format_for_backend,
    infer_data_type,
)
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.nn.backend import infer_backend
from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from modularml.core.references.execution_reference import TensorLike
    from modularml.utils.nn.backend import Backend


class SampleData(Summarizable):
    """
    Tensor-like data container for a single role across all schema domains.

    Description:
        Represents the runtime data (features, targets, and tags) associated with \
        a single role within a Batch. Each SampleData instance stores tensors or \
        array-like objects under standardized domain keys defined in \
        `SampleSchema`, including:
          - DOMAIN_FEATURES
          - DOMAIN_TARGETS
          - DOMAIN_TAGS

        This structure allows consistent access to domain data using either \
        standardized constants or convenience properties.

    Args:
        data (dict[str, Any]):
            Dictionary mapping schema domain names \
            (DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS) to tensor-like values.
        features:
            Tensor-like object representing feature data.
        targets:
            Tensor-like object representing target data.
        tags:
            Tensor-like object representing tag data.

    """

    __slots__ = ("data",)
    data: dict[str, Any]

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        sample_uuids: Any = None,
        features: Any = None,
        targets: Any = None,
        tags: Any = None,
        kind: Literal["input", "output"] = "input",
    ):
        self._kind = kind

        if data is not None:
            # Use provided dictionary directly
            self.data = dict(data)

        else:
            # Construct standardized mapping
            self.data = {}
            if sample_uuids is not None:
                self.data[DOMAIN_SAMPLE_ID] = sample_uuids
            if features is not None:
                self.data[DOMAIN_FEATURES] = features
            if targets is not None:
                self.data[DOMAIN_TARGETS] = targets
            if tags is not None:
                self.data[DOMAIN_TAGS] = tags

        self._shapes = SampleShapes(
            shapes={k: ensure_tuple_shape(v.shape[1:]) for k, v in self.data.items()},
            kind=kind,
        )

    # ================================================
    # Shape access
    # ================================================
    @property
    def shapes(self) -> SampleShapes:
        return self._shapes

    # ================================================
    # Data access
    # ================================================
    @property
    def sample_uuids(self):
        """Tensor-like data stored under DOMAIN_SAMPLE_ID."""
        return self.data.get(DOMAIN_SAMPLE_ID)

    @property
    def features(self):
        """Tensor-like data stored under DOMAIN_FEATURES."""
        return self.data.get(DOMAIN_FEATURES)

    @property
    def targets(self):
        """Tensor-like data stored under DOMAIN_TARGETS."""
        return self.data.get(DOMAIN_TARGETS)

    @property
    def tags(self):
        """Tensor-like data stored under DOMAIN_TAGS."""
        return self.data.get(DOMAIN_TAGS)

    @property
    def outputs(self):
        if self._kind != "output":
            msg = "`outputs` is only defined for SampleData produced by a model."
            raise AttributeError(msg)
        return self.features

    def get_domain_data(self, domain: str) -> TensorLike:
        """Retrieves the tensor-like data stored in the specified domain."""
        valid_attrs = ("sample_uuids", "features", "targets", "tags", "outputs")
        if domain not in valid_attrs:
            msg = f"Invalid domain: '{domain}'. Available: {valid_attrs}."
            raise KeyError(msg)
        return getattr(self, domain)

    # ================================================
    # Format conversion (inplace and copy)
    # ================================================
    def as_format(self, fmt: DataFormat):
        """
        Casts the data to the specified tensor-like DataFormat.

        This is an *in-place* conversion. If copying is needed,
        use `to_format`.

        Description:
            Performs an in-place data casting of the underlying tensors.
            Note that only the features and targets are converted; tags
            are left in the format originally defined in FeatureSet
            construction.

        Args:
            fmt (DataFormat):
                The format to cast to. `fmt` must be TensorLike
                (e.g., "torch", "tf", "np").

        """
        if not format_is_tensorlike(fmt):
            msg = f"DataFormat must be tensor-like. Received: {fmt}. Allowed formats: {_TENSORLIKE_FORMATS}."
            raise ValueError(msg)

        for k_to_convert in [DOMAIN_FEATURES, DOMAIN_TARGETS]:
            if k_to_convert not in self.data:
                continue
            self.data[k_to_convert] = convert_to_format(
                data=self.data[k_to_convert],
                fmt=fmt,
            )

    def to_format(self, fmt: DataFormat) -> SampleData:
        """
        Casts the data to the specified tensor-like DataFormat.

        This is a *non-mutating* conversion. A new copy is returned
        with the old SampleData isntance unchanged.

        Description:
            Performs a data casting on a copy of the underlying tensors.
            Note that only the features and targets are converted; tags
            are left in the format originally defined in FeatureSet
            construction.

        Args:
            fmt (DataFormat):s
                The format to cast to. `fmt` must be TensorLike
                (e.g., "torch", "tf", "np").

        """
        new_data = SampleData(data=dict(self.data), kind=self._kind)
        new_data.as_format(fmt=fmt)
        return new_data

    def as_backend(self, backend: Backend):
        """
        Casts the data to a DataFormat compatible with specified backend.

        This is an *in-place* conversion. If copying is needed,
        use `to_format`.

        Description:
            Performs an in-place data casting of the underlying tensors.
            Reformats the features and targets, but not the tags.

        """
        return self.as_format(get_data_format_for_backend(backend=backend))

    def to_backend(self, backend: Backend) -> SampleData:
        """
        Casts the data to a DataFormat compatible with specified backend.

        This is a *non-mutating* conversion. A new copy is returned
        with the old SampleData isntance unchanged.

        Description:
            Performs a data casting on a copy of the underlying tensors.
            Reformats the features and targets, but not the tags.

        """
        return self.to_format(get_data_format_for_backend(backend=backend))

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(
        self,
        row_order: Literal["attribute", "domain"] = "domain",
    ) -> list[tuple]:
        # "attribute": rows = shapes, dtype, backend
        if row_order == "attribute":
            rows: list[tuple] = []
            shape_rows = self.shapes._summary_rows()
            dtype_rows: list[tuple] = []
            backend_rows: list[tuple] = []

            domain_order = [
                DOMAIN_FEATURES,
                DOMAIN_TARGETS,
                DOMAIN_TAGS,
                DOMAIN_SAMPLE_ID,
            ]
            for d in domain_order:
                row_lbl = d
                if d == DOMAIN_FEATURES and self._kind == "output":
                    row_lbl = DOMAIN_OUTPUTS

                if d not in self.data:
                    dtype_rows.append((row_lbl, "None"))
                    continue

                v = self.data[d]
                dtype_rows.append(
                    (row_lbl, str(infer_data_type(v))),
                )
                backend_rows.append(
                    (row_lbl, str(infer_backend(v).value)),
                )

            rows.append(("shapes", shape_rows))
            rows.append(("dtype", dtype_rows))
            rows.append(("backend", backend_rows))
            return rows

        # "domain": rows = features, targets tags, sample_id
        rows: list[tuple] = []
        domain_order = [
            DOMAIN_FEATURES,
            DOMAIN_TARGETS,
            DOMAIN_TAGS,
            DOMAIN_SAMPLE_ID,
        ]
        for d in domain_order:
            row_lbl = d
            if d == DOMAIN_FEATURES and self._kind == "output":
                row_lbl = DOMAIN_OUTPUTS
            if d not in self.data:
                rows.append((row_lbl, "None"))
                continue
            v = self.data[d]
            rows.append(
                (
                    row_lbl,
                    [
                        ("shape", str(self.shapes.shapes[d])),
                        ("dtype", str(infer_data_type(v))),
                        ("backend", str(infer_backend(v).value)),
                    ],
                ),
            )

        return rows

    def __repr__(self) -> str:
        f_lbl = DOMAIN_FEATURES
        if self._kind == "output":
            f_lbl = DOMAIN_OUTPUTS
        f = self.shapes.features_shape
        t = self.shapes.targets_shape
        g = self.shapes.tags_shape
        return f"SampleData({f_lbl}={f}, targets={t}, tags={g})"


class RoleData(Mapping[str, SampleData], Summarizable):
    """
    Immutable, role-keyed container for SampleData objects.

    Description:
        RoleData is a lightweight wrapper around a mapping of role names to
        SampleData instances. It provides role introspection, attribute-style
        access, readable summaries, and a stable return type for model forward
        passes.

        When exactly one role is present, domain attributes such as `features`,
        `targets`, `tags`, and `outputs` are exposed directly for convenience.
        If multiple roles exist, direct domain access is disallowed to prevent
        ambiguity.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, SampleData]):
        if not data:
            raise ValueError("RoleData must contain at least one role.")

        for role, sd in data.items():
            if not isinstance(role, str):
                msg = f"Role keys must be str, got {type(role)}"
                raise TypeError(msg)
            if not isinstance(sd, SampleData):
                msg = f"Role '{role}' must map to SampleData, got {type(sd)}"
                raise TypeError(msg)

        self._data = dict(data)

    # ================================================
    # Mapping interface
    # ================================================
    def __getitem__(self, role: str) -> SampleData:
        return self._data[role]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ================================================
    # Properties and data access
    # ================================================
    @property
    def available_roles(self) -> list[str]:
        """List of available role names."""
        return list(self._data.keys())

    @property
    def shapes(self) -> SampleShapes:
        """
        The per-domain data shapes.

        By definition, all roles have the same shapes.
        """
        return self.get_data(role=self.available_roles[0]).shapes

    def get_data(
        self,
        role: str,
        domain: str | None = None,
    ) -> SampleData | TensorLike:
        """
        Retrieves the data stored in the specified role.

        Args:
            role (str):
                The role to retrieve data from.

            domain (str, optional):
                An optional domain within the given role to return.
                If None, the entire SampleData instance of the given
                role is returned. Defaults to None.

        Returns:
            SampleData | TensorLike

        """
        if role not in self._data:
            msg = f"Role '{role}' not found. Available roles: {self.available_roles}"
            raise KeyError(msg)
        if domain is not None:
            return self._data[role].get_domain_data(domain=domain)
        return self._data[role]

    # ================================================
    # SampleData Pass-through for Single Role
    # ================================================
    def _require_single_role(self) -> SampleData:
        """
        Return the sole SampleData if exactly one role exists.

        Raises:
            RuntimeError:
                If multiple roles are present.

        """
        if len(self._data) != 1:
            msg = (
                "Direct domain access is only supported when exactly one role "
                f"is present. Available roles: {self.available_roles}. "
                "Use `get_data` instead."
            )
            raise RuntimeError(msg)
        return next(iter(self._data.values()))

    @property
    def sample_uuids(self):
        """Tensor-like sample UUIDs (only valid if data has a single role)."""
        return self._require_single_role().sample_uuids

    @property
    def features(self):
        """Tensor-like feature data (only valid if data has a single role)."""
        return self._require_single_role().features

    @property
    def targets(self):
        """Tensor-like target data (only valid if data has a single role)."""
        return self._require_single_role().targets

    @property
    def tags(self):
        """Tensor-like tag data (only valid if data has a single role)."""
        return self._require_single_role().tags

    @property
    def outputs(self):
        """Tensor-like output data (only valid if data has a single role)."""
        return self._require_single_role().outputs

    # ================================================
    # Pseudo-attribute access
    # ================================================
    def __getattr__(self, name: str) -> SampleData:
        # Called only if attribute not found normally
        if name in self._data:
            return self._data[name]
        msg = f"{self.__class__.__name__} has no role '{name}'. Available roles: {self.available_roles}."
        raise AttributeError(msg)

    # ================================================
    # Utilities
    # ================================================
    def copy(self) -> RoleData:
        """Returns new RoleData instance with copied data."""
        new_rd = {}
        for k, v in self._data.items():
            new_rd[k] = SampleData(data=dict(v.data), kind=v._kind)
        return RoleData(data=new_rd)

    # ================================================
    # Format conversion (inplace and copy)
    # ================================================
    def as_format(self, fmt: DataFormat):
        """
        Casts all SampleData objects to the specified tensor-like DataFormat.

        This is an *in-place* conversion. If copying is needed,
        use `to_format`.

        Description:
            Performs an in-place data casting of the underlying tensors
            for every role. Only features and targets are converted;
            tags and sample IDs are left untouched.

        Args:
            fmt (DataFormat):
                The format to cast to. `fmt` must be TensorLike
                (e.g., "torch", "tf", "np").

        """
        if not format_is_tensorlike(fmt):
            msg = f"DataFormat must be tensor-like. Received: {fmt}. Allowed formats: {_TENSORLIKE_FORMATS}."
            raise ValueError(msg)

        for sd in self._data.values():
            sd.as_format(fmt=fmt)

    def to_format(self, fmt: DataFormat) -> RoleData:
        """
        Casts all SampleData objects to the specified tensor-like DataFormat.

        This is a *non-mutating* conversion. A new RoleData instance
        is returned with the original unchanged.

        Description:
            Performs data casting of the underlying tensors for every
            role on a new copy. Only features and targets are converted;
            tags and sample IDs are left untouched.

        Args:
            fmt (DataFormat):s
                The format to cast to. `fmt` must be TensorLike
                (e.g., "torch", "tf", "np").

        """
        new_rd = self.copy()
        new_rd.as_format(fmt=fmt)
        return new_rd

    def as_backend(self, backend: Backend):
        """
        Casts the data to a DataFormat compatible with the specified backend.

        This is an *in-place* conversion. If copying is needed, use `to_format`.

        Description:
            Performs an in-place data casting of the underlying tensors.
            Reformats the features and targets, but not the tags.

        """
        return self.as_format(get_data_format_for_backend(backend=backend))

    def to_backend(self, backend: Backend) -> RoleData:
        """
        Casts the data to a DataFormat compatible with the specified backend.

        This is a *non-mutating* conversion. A new copy is returned with the
        old SampleData instance unchanged.

        Description:
            Performs a data casting on a copy of the underlying tensors.
            Reformats the features and targets, but not the tags.

        """
        return self.to_format(get_data_format_for_backend(backend=backend))

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(
        self,
        row_order: Literal["attribute", "domain"] = "domain",
    ) -> list[tuple]:
        rows: list[tuple] = []

        # Available roles
        rows.append(("roles", [(r, "") for r in self.available_roles]))

        # One row per role with SampleData summary
        for role, sample_data in self._data.items():
            rows.append((role, sample_data._summary_rows(row_order=row_order)))

        return rows

    def __repr__(self) -> str:
        return f"RoleData(roles={self.available_roles})"


class SampleShapes(Summarizable):
    """
    Shape specification for all schema domains with tensor-like data.

    Description:
        Defines the per-sample tensor shapes of all schema domains
        (features, targets, tags) for one node in the model graph.
        Shapes are stored without the batch dimension, since batch size
        may vary at runtime and is not part of the static schema.

        The same shape applies to all roles associated with the node.

    Args:
        shapes (dict[str, tuple[int, ...]]):
            Dictionary mapping each schema domain name
            (DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS) to a shape
            tuple (excluding batch dimension).
        features_shape (tuple[int, ...]]):
            Tuple of ints representing feature shape.
        targets_shape (tuple[int, ...]]):
            Tuple of ints representing target shape.
        tags_shape (tuple[int, ...]]):
            Tuple of ints representing tag shape.

    """

    __slot__ = ("shapes",)
    shapes: dict[str, tuple[int, ...]]

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]] | None = None,
        *,
        features_shape: tuple[int, ...] | None = None,
        targets_shape: tuple[int, ...] | None = None,
        tags_shape: tuple[int, ...] | None = None,
        kind: Literal["input", "output"] = "input",
    ):
        self._kind = kind
        if shapes is not None:
            # Use provided dictionary directly
            self.shapes = dict(shapes)
        else:
            # Construct standardized mapping
            self.shapes = {}
            if features_shape is not None:
                self.shapes[DOMAIN_FEATURES] = features_shape
            if targets_shape is not None:
                self.shapes[DOMAIN_TARGETS] = targets_shape
            if tags_shape is not None:
                self.shapes[DOMAIN_TAGS] = tags_shape

    def __getitem__(self, domain: str) -> tuple[int, ...]:
        """The shape tuple corresponding to a given schema domain."""
        return self.shapes[domain]

    def __repr__(self):
        return f"SampleShapes({self.shapes})"

    @property
    def features_shape(self) -> tuple[int, ...] | None:
        """Shape tuple for the DOMAIN_FEATURES."""
        if DOMAIN_FEATURES in self.shapes:
            return self.shapes[DOMAIN_FEATURES]
        return None

    @property
    def outputs_shape(self) -> tuple[int, ...] | None:
        """Shape tuple for the DOMAIN_OUTPUTS."""
        if self._kind != "output":
            msg = (
                "`outputs_shape` is only defined for SampleShapes produced by a model."
            )
            raise AttributeError(msg)
        return self.features_shape

    @property
    def targets_shape(self) -> tuple[int, ...] | None:
        """Shape tuple for the DOMAIN_TARGETS domain."""
        if DOMAIN_TARGETS in self.shapes:
            return self.shapes[DOMAIN_TARGETS]
        return None

    @property
    def tags_shape(self) -> tuple[int, ...] | None:
        """Shape tuple for the DOMAIN_TAGS domain."""
        if DOMAIN_TAGS in self.shapes:
            return self.shapes[DOMAIN_TAGS]
        return None

    def _summary_rows(self) -> list[tuple]:
        rows: list[tuple] = []

        domain_priority = [
            DOMAIN_FEATURES,
            DOMAIN_TARGETS,
            DOMAIN_TAGS,
        ]
        for d in domain_priority:
            row_lbl = d
            if d == DOMAIN_FEATURES and self._kind == "output":
                row_lbl = DOMAIN_OUTPUTS
            if d not in self.shapes:
                rows.append((row_lbl, "None"))
                continue

            rows.append((row_lbl, str(self.shapes[d])))

        return rows
