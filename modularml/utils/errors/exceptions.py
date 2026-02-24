"""Custom exception hierarchy for ModularML."""

from modularml.utils.nn.backend import Backend


class ModularMLError(Exception):
    """Base class for all ModularML-specific exceptions."""


class BackendNotSupportedError(ModularMLError):
    """Raised when an operation is called on a backend that is not supported."""

    def __init__(
        self,
        backend: Backend,
        method: str | None = None,
        message: str | None = None,
    ):
        """
        Initialize error with backend context.

        Args:
            backend (Backend): Backend that triggered the error.
            method (str | None, optional): Method name where the error occurred.
            message (str | None, optional): Custom message override.

        """
        if message is None:
            if method is None:
                message = f"Unsupported backend '{backend}'."
            else:
                message = f"Unsupported backend '{backend}' in method `{method}`"
        super().__init__(message)


class BackendMismatchError(ModularMLError):
    """
    Raised when a component receives input or configuration with the wrong backend.

    Attributes:
        expected (str): Backend expected by the component.
        received (str): Backend actually provided.

    """

    def __init__(self, expected: str, received: str, message: str | None = None):
        """
        Initialize mismatch error.

        Args:
            expected (str): Backend name required.
            received (str): Backend name provided.
            message (str | None, optional): Custom message override.

        """
        if message is None:
            message = f"Expected backend '{expected}', but got '{received}'."
        super().__init__(message)
        self.expected = expected
        self.received = received


class ActivationError(ModularMLError):
    """Raised when error occurs with Activation class."""

    def __init__(self, message: str | None = None):
        """Initialize activation error with optional message."""
        if message is None:
            message = "Error with Activation."
        super().__init__(message)


class OptimizerError(ModularMLError):
    """Raised when error occurs with Optimizer class."""

    def __init__(self, message: str | None = None):
        """Initialize optimizer error with optional message."""
        if message is None:
            message = "Error with Optimizer."
        super().__init__(message)


class OptimizerNotSetError(OptimizerError):
    """Raised when training is called on a ModelStage with no Optimizer."""

    def __init__(self, method: str | None = None, message: str | None = None):
        """
        Initialize optimizer-not-set error.

        Args:
            method (str | None, optional): Method lacking optimizer.
            message (str | None, optional): Custom message override.

        """
        if message is None:
            message = (
                "Missing Optimizer."
                if method is None
                else f"Missing Optimizer in method `{method}`."
            )
        super().__init__(message)


class LossError(ModularMLError):
    """Raised when error occurs with Loss class."""

    def __init__(self, message: str | None = None):
        """Initialize loss error with optional message."""
        if message is None:
            message = "Error with Loss."
        super().__init__(message)


class LossNotSetError(LossError):
    """Raised when loss is None but required."""

    def __init__(self, method: str | None = None, message: str | None = None):
        """
        Initialize loss-not-set error.

        Args:
            method (str | None, optional): Method missing loss configuration.
            message (str | None, optional): Custom message override.

        """
        if message is None:
            message = (
                "Missing AppliedLoss."
                if method is None
                else f"Missing AppliedLoss in method `{method}`."
            )
        super().__init__(message)


class GraphNodeError(ModularMLError):
    """Base exception for graph-node issues."""

    def __init__(self, message: str | None = None):
        """Initialize graph-node error with optional message."""
        if message is None:
            message = "Error with GraphNode"
        super().__init__(message)


class GraphNodeInputError(GraphNodeError):
    """Raised when invalid inputs are provided to a graph node."""

    def __init__(self, message: str | None = None):
        """Initialize input error for graph nodes."""
        if message is None:
            message = "Error with GraphNode input."
        super().__init__(message)


class GraphNodeOutputError(GraphNodeError):
    """Raised when invalid outputs are produced by a graph node."""

    def __init__(self, message: str | None = None):
        """Initialize output error for graph nodes."""
        if message is None:
            message = "Error with GraphNode output."
        super().__init__(message)


class ModelStageError(ModularMLError):
    """Raised when error occurs within ModelStage class."""

    def __init__(self, message: str | None = None):
        """Initialize model-stage error with optional message."""
        if message is None:
            message = "Error with ModelStage."
        super().__init__(message)


class ModelStageInputError(ModelStageError):
    def __init__(self, message: str | None = None):
        """Initialize model-stage input error."""
        if message is None:
            message = "Error with ModelStage input."
        super().__init__(message)


class FeatureSetError(ModularMLError):
    """Base exception for FeatureSet-related issues."""

    def __init__(self, message: str | None = None):
        """Initialize featureset error with optional message."""
        if message is None:
            message = "Error with FeatureSet."
        super().__init__(message)


class SampleLoadError(FeatureSetError):
    """Raised when FeatureSet samples cannot be loaded."""

    def __init__(self, message: str | None = None):
        """Initialize sample-load error."""
        if message is None:
            message = "Failed to load Samples."
        super().__init__(message)


class SplitOverlapWarning(UserWarning):
    """Warning emitted when FeatureSet splits overlap unexpectedly."""


class NotInvertibleError(ModularMLError):
    """Raised when a node's outputs cannot be inverse-transformed (merged sources, incompatible shapes, etc.)."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Failed to apply inverse_transform."
        super().__init__(message)


class ShapeSpecError(ModularMLError):
    """Raised when errors occur relating to invalid ShapeSpecs."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Invalid ShapeSpec."
        super().__init__(message)
