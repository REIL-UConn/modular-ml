from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modularml.core.data.sample_data import RoleData, SampleShapes
from modularml.utils.data.data_format import (
    DataFormat,
    get_data_format_for_backend,
    infer_data_type,
)
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

    from modularml.core.data.batch import Batch
    from modularml.core.data.sample_data import SampleData
    from modularml.core.references.execution_reference import TensorLike
    from modularml.utils.nn.backend import Backend


@dataclass(frozen=True)
class Batch(Summarizable):
    """
    Immutable, role-structured tensor container produced from a single BatchView.

    A Batch represents one sampler's worth of data for a single
    execution step. It contains no graph or node semantics.
    """

    batch_size: int

    # Core role-based storage
    role_data: RoleData
    shapes: SampleShapes
    role_weights: Mapping[str, NDArray[np.float32]]
    role_masks: Mapping[str, NDArray[np.int8]]

    # ================================================
    # Validation
    # ================================================
    def __post_init__(self):
        # Cast role_data to RoleData (if not already)
        if not isinstance(self.role_data, RoleData):
            object.__setattr__(self, "role_data", RoleData(data=self.role_data))

        # Ensure shapes is set
        if self.shapes is None or not isinstance(self.shapes, SampleShapes):
            s_data: SampleData = self.role_data.get_data(
                role=self.role_data.available_roles[0],
            )
            object.__setattr__(self, "shapes", s_data.shapes)

        # Validate shapes and role keys
        avail_roles = set(self.role_data.available_roles)
        if not avail_roles:
            raise ValueError("MaterializedBatch must contain at least one role.")

        if set(self.role_weights) != avail_roles:
            raise ValueError("`role_weights` keys must match `role_data` roles.")

        for role in avail_roles:
            sample_data: SampleData = self.role_data.get_data(role=role)
            weights = self.role_weights[role]
            mask = self.role_masks[role]

            if weights.shape != (self.batch_size,):
                msg = f"role_weights['{role}'] has shape {weights.shape}, expected ({self.batch_size},)"
                raise ValueError(msg)
            if mask.shape != (self.batch_size,):
                msg = f"role_masks['{role}'] has shape {mask.shape}, expected ({self.batch_size},)"
                raise ValueError(msg)

            # Validate batch dimension consistency
            for domain, tensor in sample_data.data.items():
                if tensor is not None and tensor.shape[0] != self.batch_size:
                    msg = (
                        f"{role}.{domain} has leading dimension {tensor.shape[0]}, "
                        f"expected batch_size={self.batch_size}"
                    )
                    raise ValueError(msg)

    # ================================================
    # Data access
    # ================================================
    @property
    def available_roles(self) -> list[str]:
        return self.role_data.available_roles

    def get_data(
        self,
        role: str | None = None,
        domain: str | None = None,
    ) -> RoleData | SampleData | TensorLike:
        """
        Retrieves the data stored in this batch.

        Args:
            role (str, optional):
                An optional role name within the batch data to return.
                If None, the entire RoleData instance is returned.
                Note that `domain` is ignored if None.
                Defaults to None.

            domain (str, optional):
                An optional domain within the given role to return.
                If specified, `role` must be defined. If None,
                the entire SampleData instance (per role) is returned.
                Defaults to None.

        Returns:
            RoleData | SampleData | TensorLike

        """
        if role is None:
            return self.role_data
        if domain is None:
            return self.role_data.get_data(role=role)
        return self.role_data.get_data(role=role, domain=domain)

    # ================================================
    # Pseudo-attribute access
    # ================================================
    def __getattr__(self, name: str):
        # Called only if attribute not found normally
        if name in self.role_data.available_roles:
            return self.get_data(role=name)
        msg = f"{self.__class__.__name__} has no attribute '{name}'. Available roles: {self.available_roles}."
        raise AttributeError(msg)

    # ================================================
    # Format conversion (inplace and copy)
    # ================================================
    def as_format(self, fmt: DataFormat):
        """
        Casts all tensor-like data in this Batch to the specified DataFormat.

        This is an *in-place* conversion. If copying is needed,
        use `to_format`.

        Description:
            Performs an in-place data casting of the underlying tensors
            for all roles. Only SampleData (features / targets / outputs)
            are converted. Weights and masks are left untouched.

        Args:
            fmt (DataFormat):
                Target tensor-like format (e.g., "torch", "tf", "np").

        """
        self.role_data.as_format(fmt)

    def to_format(self, fmt: DataFormat) -> Batch:
        """
        Casts all tensor-like data in this Batch to the specified DataFormat.

        This is a *non-mutating* conversion. A new Batch instance
        is returned with the original unchanged.

        Description:
            Performs data casting on a copy of the underlying tensors
            for all roles. Only SampleData (features / targets / outputs)
            are converted. Weights and masks are copied, but not re-formatted.

        Args:
            fmt (DataFormat):
                Target tensor-like format (e.g., "torch", "tf", "np").

        """
        new_rd = self.role_data.to_format(fmt)
        new_rd.as_format(fmt=fmt)
        return Batch(
            batch_size=self.batch_size,
            role_data=new_rd,
            shapes=self.shapes,
            role_weights=dict(self.role_weights),
            role_masks=dict(self.role_masks),
        )

    def as_backend(self, backend: Backend):
        """
        Casts tensor-like data to be compatible with a specified backend.

        This is an *in-place* conversion. If copying is needed,
        use `to_format`.

        Description:
            Performs an in-place data casting of the underlying tensors
            for all roles. Only SampleData (features / targets / outputs)
            are converted. Weights and masks are left untouched.
        """
        return self.as_format(get_data_format_for_backend(backend=backend))

    def to_backend(self, backend: Backend) -> Batch:
        """
        Casts tensor-like data to be compatible with a specified backend.

        This is a *non-mutating* conversion. A new copy is returned with the
        old Batch instance unchanged.

        Description:
            Performs data casting on a copy of the underlying tensors
            for all roles. Only SampleData (features / targets / outputs)
            are converted. Weights and masks are copied, but not re-formatted.

        """
        return self.to_format(get_data_format_for_backend(backend=backend))

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        # One row per role with SampleData summary
        r_rows = []
        for role, sample_data in self.role_data.items():
            r_rows.append((role, sample_data._summary_rows(row_order="domain")))

        rows = [
            ("batch_size", self.batch_size),
            ("roles", r_rows),
            # *self.role_data._summary_rows(row_order="domain"),
        ]
        rw_rows = []
        for k, v in self.role_weights.items():
            rw_rows.append(
                (
                    k,
                    [
                        ("shape", str(ensure_tuple_shape(v.shape))),
                        ("dtype", str(infer_data_type(v))),
                    ],
                ),
            )

        rm_rows = []
        for k, v in self.role_masks.items():
            rm_rows.append(
                (
                    k,
                    [
                        ("shape", str(ensure_tuple_shape(v.shape))),
                        ("dtype", str(infer_data_type(v))),
                    ],
                ),
            )

        rows.append(("role_weights", rw_rows))
        rows.append(("role_masks", rm_rows))
        return rows

    def __repr__(self) -> str:
        return f"Batch(batch_size={self.batch_size}, roles={self.available_roles})"
