"""Scaling helpers for reversing FeatureSet and RoleData transforms."""

from modularml.core.data.batch import Batch
from modularml.core.data.sample_data import RoleData, SampleData
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_TARGETS,
    REP_TRANSFORMED,
)
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.topology.graph_node import GraphNode
from modularml.utils.data.data_format import DataFormat
from modularml.utils.topology.graph_search_utils import (
    find_upstream_featuresets,
    is_tail_node,
)


def unscale_sample_data(data: SampleData, from_node: str | GraphNode) -> SampleData:
    """
    Inverse-scale sample data using the upstream FeatureSet of a graph node.

    Description:
        Reverses scaling applied to SampleData using the scaler history of the single
        FeatureSet feeding into the producing graph node. Targets are always unscaled
        when possible, while features are only unscaled if the node is a tail node
        (interpreted as prediction outputs).

    Args:
        data (SampleData):
            SampleData containing scaled features, targets, and tags.
        from_node (str | GraphNode):
            Graph node (or label) that produced the data and determines the
            FeatureSet used for inverse scaling.

    Returns:
        SampleData: New SampleData with unscaled targets and features (when applicable).

    Raises:
        TypeError: If `data` or `from_node` types are invalid.
        ValueError: If no upstream FeatureSet exists.
        RuntimeError: If multiple upstream FeatureSets are found.

    """
    if not isinstance(data, SampleData):
        msg = f"Invalid `data` type. Expected SampleData. Received: {type(data)}."
        raise TypeError(msg)

    # Get FeatureSet that fed into the producing node
    graph_node = None
    if isinstance(from_node, GraphNode):
        graph_node = from_node
    elif isinstance(from_node, str):
        exp_ctx = ExperimentContext.get_active()
        graph_node = exp_ctx.get_node(val=from_node, enforce_type="GraphNode")
    else:
        msg = (
            "Invalid `from_node` type. Expected string or GraphNode. "
            f"Received: {type(from_node)}."
        )
        raise TypeError(msg)

    # Validate only one upstream FeatureSet
    fsrs = find_upstream_featuresets(graph_node)
    if not fsrs:
        msg = f"No upstream FeatureSets found for node '{graph_node.label}'."
        raise ValueError(msg)
    if len(fsrs) > 1:
        msg = (
            f"Multiple upstream FeatureSets for node '{graph_node.label}'. "
            "Unscaling of compound data is not supported."
        )
        raise RuntimeError(msg)
    fsr = fsrs[0]
    fs = fsr.resolve().source

    # Ensure data is np arrays
    sd = data.to_format(fmt=DataFormat.NUMPY)

    # Unscale target columns (using upstream fs)
    ks = [
        col.replace(f"{DOMAIN_TARGETS}.", "").replace(f".{REP_TRANSFORMED}", "")
        for col in getattr(fsr, DOMAIN_TARGETS)
    ]
    target_data = fs.unscale_data_for_cols(
        data=sd.get_domain_data(domain=DOMAIN_TARGETS),
        domain=DOMAIN_TARGETS,
        columns=ks,
    )

    # Unscaling of SampleData.targets is always possible (if no merge nodes)
    # Unscaling of .features/.outputs is only support if is a tail node (outputs = predictions)
    if is_tail_node(node=graph_node):
        feature_data = fs.unscale_data_for_cols(
            data=sd.get_domain_data(domain=DOMAIN_FEATURES),
            domain=DOMAIN_TARGETS,
            columns=ks,
        )
    else:
        feature_data = sd.features

    # Build new SampleData
    return SampleData(
        sample_uuids=sd.sample_uuids,
        features=feature_data,
        targets=target_data,
        tags=sd.tags,
        kind=sd._kind,
    )


def unscale_role_data(data: RoleData, from_node: str | GraphNode) -> RoleData:
    """
    Inverse-scale role-based data using the upstream FeatureSet of a graph node.

    Description:
        Applies inverse scaling to all roles in the RoleData by unscaling each
        contained SampleData using the FeatureSet feeding into the producing node.

    Args:
        data (RoleData):
            RoleData containing scaled SampleData for one or more roles.
        from_node (str | GraphNode):
            Graph node (or label) that produced the data and determines the
            FeatureSet used for inverse scaling.

    Returns:
        RoleData: RoleData with unscaled data for all roles.

    Raises:
        TypeError: If `data` is not a :class:`RoleData`.

    """
    if not isinstance(data, RoleData):
        msg = f"Invalid `data` type. Expected RoleData. Received: {type(data)}."
        raise TypeError(msg)

    new_role_data = {}
    for role in data.available_roles:
        sd: SampleData = data.get_data(role=role)
        new_role_data[role] = unscale_sample_data(data=sd, from_node=from_node)
    return RoleData(data=new_role_data)


def unscale_batch_data(data: Batch, from_node: str | GraphNode) -> Batch:
    """
    Inverse-scale batch data using the upstream FeatureSet of a graph node.

    Description:
        Reverses scaling for all role-based data contained in a Batch by applying
        inverse scaling to each underlying RoleData entry.

    Args:
        data (Batch):
            Batch containing scaled role-based sample data.
        from_node (str | GraphNode):
            Graph node (or label) that produced the data and determines the
            FeatureSet used for inverse scaling.

    Returns:
        Batch: Batch with unscaled features and targets for all roles.

    Raises:
        TypeError: If `data` is not a :class:`Batch`.

    """
    if not isinstance(data, Batch):
        msg = f"Invalid `data` type. Expected Batch. Received: {type(data)}."
        raise TypeError(msg)

    new_role_data = unscale_role_data(data=data.role_data, from_node=from_node)

    return Batch(
        batch_size=data.batch_size,
        role_data=new_role_data,
        shapes=data.shapes,
        role_weights=data.role_weights,
        role_masks=data.role_masks,
    )


def unscale_data(
    data: SampleData | RoleData | Batch,
    from_node: str | GraphNode,
) -> SampleData | RoleData | Batch:
    """
    Inverse-scale sampled data using the upstream FeatureSet of a graph node.

    Description:
        Dispatches inverse scaling based on input type, supporting SampleData,
        RoleData, and Batch. Scaling is reversed using the scaler history of the
        single FeatureSet feeding into the producing node.

    Args:
        data (SampleData | RoleData | Batch):
            Scaled data to inverse-scale.
        from_node (str | GraphNode):
            Graph node (or label) that produced the data and determines the
            FeatureSet used for inverse scaling.

    Returns:
        SampleData | RoleData | Batch: Instance of the same type with unscaled values.

    """
    if isinstance(data, SampleData):
        return unscale_sample_data(data=data, from_node=from_node)
    if isinstance(data, RoleData):
        return unscale_role_data(data=data, from_node=from_node)
    if isinstance(data, Batch):
        return unscale_batch_data(data=data, from_node=from_node)
    msg = (
        "Invalid `data` type. Data must be one of SampleData, RoleData, or Batch. "
        f"Received: {type(data)}."
    )
    raise TypeError(msg)
