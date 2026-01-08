from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

from modularml.context.resolution_context import ResolutionContext
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    REP_RAW,
    T_ALL_DOMAINS,
    T_ALL_REPS,
)
from modularml.core.references.experiment_reference import ExperimentNodeReference, ExperimentReference, ResolutionError
from modularml.utils.data.pyarrow_data import resolve_column_selectors
from modularml.utils.logging.warnings import warn

if TYPE_CHECKING:
    from modularml.context.experiment_context import ExperimentContext
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView


@dataclass(frozen=True)
class FeatureSetReference(ExperimentNodeReference):
    """Declarative reference to a subset of columns from a FeatureSet."""

    # ExperimentNode
    node_label: str | None = None
    node_id: str | None = None

    # FeatureSet-specific
    features: tuple[str, ...] | None = None
    targets: tuple[str, ...] | None = None
    tags: tuple[str, ...] | None = None

    def resolve(self, ctx: ResolutionContext | None = None) -> FeatureSetView:
        """Resolves this reference to a FeatureSetView instance."""
        return super().resolve(ctx=ctx)

    def _resolve_experiment(self, experiment: ExperimentContext) -> FeatureSetView:
        from modularml.core.data.featureset import FeatureSet

        # Get FeatureSet node
        node_ref = ExperimentNodeReference(
            node_label=self.node_label,
            node_id=self.node_id,
        )
        node = node_ref._resolve_experiment(experiment=experiment)
        if not isinstance(node, FeatureSet):
            msg = f"Resolved node is not a FeatureSet. Received: {type(node)}."
            raise ResolutionError(msg)

        # If features, targets, tags are all None -> use all columns
        if all(x is None for x in [self.features, self.targets, self.tags]):
            return node.to_view()

        # Perform column-wise filtering
        all_columns = node.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        selected = resolve_column_selectors(
            all_columns=all_columns,
            features=self.features,
            targets=self.targets,
            tags=self.tags,
            include_all_if_empty=False,
        )
        if not any(selected.values()):
            raise ResolutionError("FeatureSetReference resolved to zero columns")

        return node.select(
            features=list(selected[DOMAIN_FEATURES]),
            targets=list(selected[DOMAIN_TARGETS]),
            tags=list(selected[DOMAIN_TAGS]),
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
            "features": self.features,
            "targets": self.targets,
            "tags": self.tags,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSetReference:
        """Reconstructs the reference from config."""
        return cls(**config)


@dataclass(frozen=True)
class FeatureSetSplitReference(ExperimentReference):
    """Declarative reference to a subset of columns from a FeatureSet."""

    # Split-specific
    split_name: str

    # ExperimentNode
    node_label: str | None = None
    node_id: str | None = None

    def resolve(self, ctx: ResolutionContext | None = None) -> FeatureSetView:
        """Resolves this reference to a view of the reference FeatureSet split instance."""
        return super().resolve(ctx=ctx)

    def _resolve_experiment(self, experiment: ExperimentContext) -> FeatureSetView:
        from modularml.core.data.featureset import FeatureSet

        # Get FeatureSet node
        node_ref = ExperimentNodeReference(
            node_label=self.node_label,
            node_id=self.node_id,
        )
        node = node_ref._resolve_experiment(experiment=experiment)
        if not isinstance(node, FeatureSet):
            msg = f"Resolved node is not a FeatureSet. Received: {type(node)}."
            raise ResolutionError(msg)

        # Validate split exists
        if self.split_name not in node.available_splits:
            msg = (
                f"Split '{self.split_name}' does not exist in FeatureSet '{node.label}'. "
                f"Available splits: {node.available_splits}"
            )
            raise ResolutionError(msg)

        return node.get_split(split_name=self.split_label)

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
            "split_name": self.split_name,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSetReference:
        """Reconstructs the reference from config."""
        return cls(**config)


@dataclass(frozen=True)
class FeatureSetColumnReference(ExperimentReference):
    """Reference to a single column of a FeatureSet."""

    # FeatureSet-specific (single column)
    domain: str
    key: str
    rep: str

    # ExperimentNode
    node_label: str | None = None
    node_id: str | None = None

    def __post_init__(self):
        # Validate domain
        valid_ds = [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS, DOMAIN_SAMPLE_ID]
        if self.domain not in valid_ds:
            msg = f"Domain must be one of: {valid_ds}. Received: {self.domain}."
            raise ValueError(msg)

    def resolve(self, ctx: ResolutionContext | None = None) -> FeatureSetView:
        """Resolves this single-column reference to a FeatureSetView instance."""
        return super().resolve(ctx=ctx)

    def _resolve_experiment(self, experiment: ExperimentContext) -> FeatureSetView:
        # Convert column attributes to fully-qualified column
        col = f"{self.domain}.{self.key}.{self.rep}"

        # Resolve FeatureSetView
        fs_ref = FeatureSetReference(
            node_id=self.node_id,
            node_label=self.node_label,
            **{self.domain: col},  # attach domain = single column
        )
        fsv = fs_ref._resolve_experiment(experiment=experiment)

        return fsv

    @classmethod
    def from_string(
        cls,
        val: str,
        *,
        experiment: ExperimentContext,
        known_attrs: dict[str, str] | None = None,
    ) -> FeatureSetColumnReference:
        """
        Parse a string into a FeatureSetColumnReference.

        Supported forms (order-agnostic):
            - FeatureSet.features.key.rep
            - FeatureSet.key.rep
            - FeatureSet.key
            - features.key.rep        (only if exactly one FeatureSet exists)
            - key.rep                 (same constraint)
            - key                     (same constraint)

        Resolution rules:
            - domain inferred from FeatureSet schema if missing
            - rep defaults to REP_RAW if ambiguous (with warning)

        Args:
            val (str):
                String vlaue to parsed in a true reference.
            experiment (ExperimentContext):
                Context of active nodes.
            known_attrs (dict[str, str], optional):
                Known attributes can be provided. They will be used to supplement
                any attributes parsed from `x`.

        Returns:
            FeatureSetColumnReference

        """
        parts = val.split(".")

        # Collect known FeatureSet labels
        fs_nodes = experiment.available_featuresets

        # Known context
        known_domains = set(get_args(T_ALL_DOMAINS))
        known_reps = set(get_args(T_ALL_REPS))

        # Parsed fields
        parsed: dict[str, str | None] = {
            "domain": None,
            "key": None,
            "rep": None,
            "node_label": None,
            "node_id": None,
        }
        # Load with known attributes (if any)
        if known_attrs is not None:
            for k, v in known_attrs.items():
                if k not in parsed:
                    msg = f"Invalid attribute in `known_attr` '{k}'"
                    raise ValueError(msg)
                parsed[k] = v

        unmatched: list[str] = []

        # First pass: explicit matches
        for p in parts:
            # Node label
            if p in fs_nodes:
                if parsed["node_label"] and parsed["node_label"] != p:
                    msg = f"Multiple FeatureSet nodes found in '{val}' and `known_attr`: {p} != {parsed['node_label']}."
                    raise ResolutionError(msg)
                parsed["node_label"] = p

            # Domain
            elif p in known_domains:
                if parsed["domain"] and parsed["domain"] != p:
                    msg = f"Multiple domains found in '{val}' and `known_attr`: {p} != {parsed['domain']}."
                    raise ResolutionError(msg)
                parsed["domain"] = p

            # Rep
            elif p in known_reps:
                if parsed["rep"] and parsed["rep"] != p:
                    msg = f"Multiple reps found in '{val}' and `known_attr`: {p} != {parsed['rep']}."
                    raise ResolutionError(msg)
                parsed["rep"] = p

            # Record as unmatched
            else:
                unmatched.append(p)

        # Is no parsed node id, can only infer if only 1 featureset exists
        if parsed["node_id"] is None and parsed["node_label"] is None:
            if len(fs_nodes) == 1:
                parsed["node_label"] = fs_nodes[0]
            else:
                msg = f"FeatureSet not specified and cannot be inferred in '{val}'. Available FeatureSets: {fs_nodes}"
                raise ResolutionError(msg)

        # Resolve FeatureSet (need node_id or node_label)
        fs_ref = FeatureSetReference(
            node_id=parsed["node_id"],
            node_label=parsed["node_label"],
        )
        fs: FeatureSet = fs_ref.resolve(
            ctx=ResolutionContext(experiment=experiment),
        ).source

        # Resolve domain and column
        if parsed["domain"] is None and parsed["key"] is not None:
            # Get potential domains, given the column key
            all_keys = fs.get_all_keys(
                include_domain_prefix=True,
                include_rep_suffix=False,
            )
            avail_parsed: list[tuple[str, str | None]] = []
            for k in all_keys:
                if "." in k:
                    d, col = k.split(".", maxsplit=1)
                else:
                    d, col = k, None
                if col == parsed["key"]:
                    avail_parsed.append((d, col))

            # If only one possible avilable -> use it
            if len(avail_parsed) == 1:
                parsed["domain"] = avail_parsed[0][0]

            # Otherwise, check against unmatched values
            elif not unmatched:
                msg = f"No domain found in '{val}'."
                raise ResolutionError(msg)

            else:
                unmatched_cands = [(d, v) for k in unmatched if k in [a[1] for a in avail_parsed]]
                if len(unmatched_cands) != 1:
                    msg = f"Could not uniquely identify a domain in '{val}'. Possible candidates: {unmatched_cands}."
                    raise ResolutionError(msg)

                parsed["domain"] = unmatched_cands[0][0]

        # Resolve column key
        if parsed["key"] is None:
            if not unmatched:
                msg = f"No column key found in '{val}'."
                raise ResolutionError(msg)

            # Get available columns to match to
            if parsed["domain"]:
                all_keys = fs.collection._get_domain_keys(
                    domain=parsed["domain"],
                    include_domain_prefix=True,
                    include_rep_suffix=False,
                )
            else:
                all_keys = fs.get_all_keys(
                    include_domain_prefix=True,
                    include_rep_suffix=False,
                )
                all_keys.remove(DOMAIN_SAMPLE_ID)
            # Parse available into domain, colummn key
            avail_parsed: list[tuple[str, str | None]] = []
            for k in all_keys:
                if "." in k:
                    d, col = k.split(".", maxsplit=1)
                else:
                    d, col = k, None
                avail_parsed.append((d, col))

            candidates = [(d, k) for k in unmatched if k in [a[1] for a in avail_parsed]]
            if len(candidates) != 1:
                msg = f"Could not uniquely identify column key in '{val}'. Possible candidates: {candidates}."
                raise ResolutionError(msg)

            parsed["domain"] = candidates[0][0]
            parsed["key"] = candidates[0][1]
            if parsed["domain"] in unmatched:
                unmatched.remove(parsed["domain"])
            unmatched.remove(parsed["key"])

        # Resolve rep
        if parsed["rep"] is None:
            # Check available reps on column
            avail_reps = fs.collection._get_rep_keys(
                domain=parsed["domain"],
                key=parsed["key"],
            )

            # Check if `unmatched` vals are in `avail_reps`
            if unmatched:
                candidates = [v for v in unmatched if v in avail_reps]
                if len(candidates) != 1:
                    msg = f"Could not uniquely identify rep in '{val}'. Possible candidates: {candidates}."
                    raise ResolutionError(msg)

                parsed["rep"] = candidates[0]
                unmatched.remove(candidates[0])

            # If no unmatched, default to RAW_REP but raise warning
            elif len(avail_reps) == 1:
                parsed["rep"] = avail_reps[0]

            elif REP_RAW in avail_reps:
                msg = f" Multiple possible `reps` for '{val}'. Selecting the default representation: '{REP_RAW}'."
                warn(msg, category=UserWarning, stacklevel=2)
                parsed["rep"] = REP_RAW

            else:
                msg = f"Failed to identify rep in '{val}'."
                raise ResolutionError(msg)

        # Final validation
        if parsed["domain"] == DOMAIN_SAMPLE_ID:
            if parsed["key"] is not None and parsed["rep"] is not None:
                msg = f"Failed to resolved '{val}' into column reference."
                raise ResolutionError(msg)
        else:
            fq = f"{parsed['domain']}.{parsed['key']}.{parsed['rep']}"
            if fq not in fs.get_all_keys(
                include_domain_prefix=True,
                include_rep_suffix=True,
            ):
                msg = f"Failed to resolved '{val}' into column reference."
                raise ResolutionError(msg)

        return cls(
            node_label=fs.label,
            node_id=fs.node_id,
            domain=parsed["domain"],
            key=parsed["key"],
            rep=parsed["rep"],
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
            "domain": self.domain,
            "key": self.key,
            "rep": self.rep,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSetReference:
        """Reconstructs the reference from config."""
        return cls(**config)
