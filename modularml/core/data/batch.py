from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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

    from numpy.typing import NDArray

    from modularml.core.data.batch import Batch
    from modularml.core.data.sample_data import SampleData
    from modularml.core.references.execution_reference import TensorLike
    from modularml.utils.nn.backend import Backend


@dataclass(frozen=True)
class Batch(Summarizable):
    """
    Immutable, role-structured tensor container produced from a single BatchView.

    Description:
        A Batch represents one sampler's worth of data for a single execution step.
        It stores role-structured SampleData along with per-role weights and masks.

        When exactly one role is present, domain attributes such as `features`,
        `targets`, `tags`, `outputs`, and `sample_uuids` are exposed directly via
        delegation to the underlying RoleData for convenience. If multiple roles
        exist, direct domain access is disallowed to prevent ambiguity.
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
    # SampleData Pass-through for Single Role
    # ================================================
    @property
    def sample_uuids(self):
        """
        Tensor-like sample UUIDs for the batch.

        Only valid when exactly one role is present. Raises an error if
        multiple roles exist.
        """
        return self.role_data.sample_uuids

    @property
    def features(self):
        """
        Tensor-like feature data for the batch.

        Only valid when exactly one role is present. Raises an error if
        multiple roles exist.
        """
        return self.role_data.features

    @property
    def targets(self):
        """
        Tensor-like target data for the batch.

        Only valid when exactly one role is present. Raises an error if
        multiple roles exist.
        """
        return self.role_data.targets

    @property
    def tags(self):
        """
        Tensor-like tag data for the batch.

        Only valid when exactly one role is present. Raises an error if
        multiple roles exist.
        """
        return self.role_data.tags

    @property
    def outputs(self):
        """
        Tensor-like output (prediction) data for the batch.

        Only valid when exactly one role is present and the underlying
        SampleData represents model outputs.
        """
        return self.role_data.outputs

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
    # Concatenation
    # ================================================
    def concat(
        self,
        *others: Batch,
        fmt: DataFormat | None = None,
    ) -> Batch:
        """
        Concatenate this Batch with one or more others along the batch axis.

        Description:
            Concatenates Batch instances by stacking role-based SampleData,
            role weights, and role masks along the leading (batch) dimension.
            All batches must share identical role structure and shapes.

            Tensor concatenation behavior follows SampleData.concat semantics.

        Args:
            *others (Batch):
                Additional Batch instances to concatenate.
            fmt (DataFormat | None, optional):
                Target format to use for concatenation. If None, the backend is
                inferred from the inputs when possible.

        Returns:
            Batch:
                A new Batch instance containing concatenated data.

        Raises:
            TypeError:
                If any input is not a Batch instance.
            ValueError:
                If Batch instances are structurally incompatible.

        """

        def _concat_all(*batches: Batch) -> Batch:
            # Validate types
            for b in batches:
                if not isinstance(b, Batch):
                    msg = f"Expected Batch, got {type(b)}."
                    raise TypeError(msg)

            # Concate role data (perform validation checks)
            new_rd = RoleData.concat(*[b.role_data for b in batches], fmt=fmt)

            # Concat weights and mask
            new_rweights = {}
            new_rmasks = {}
            for role in new_rd.available_roles:
                new_rweights[role] = np.concatenate(
                    [b.role_weights[role] for b in batches],
                    axis=0,
                )
                new_rmasks[role] = np.concatenate(
                    [b.role_masks[role] for b in batches],
                    axis=0,
                )
            return Batch(
                batch_size=sum(b.batch_size for b in batches),
                role_data=new_rd,
                shapes=new_rd.shapes,
                role_weights=new_rweights,
                role_masks=new_rmasks,
            )

        # Support RoleData.concat(rd1, rd2, ...)
        if isinstance(self, type):
            return _concat_all(*others)

        # Support rd1.concat(rd2, rd3, ...)
        return _concat_all(self, *others)

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
