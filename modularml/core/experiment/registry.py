from modularml.core.io.symbol_registry import symbol_registry
# from modularml.core.io.conventions import SerializationKind, kind_registry

from modularml.core.experiment.callback import Callback, CallbackResult
from modularml.callbacks import callback_naming_fn, callback_registry

from modularml.core.experiment.phases.phase import ExperimentPhase
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.phases.phase_result import PhaseResults


def register_builtin():
    # Callbacks
    symbol_registry.register_builtin_class(
        key="Callback",
        cls=Callback,
    )
    symbol_registry.register_builtin_class(
        key="CallbackResult",
        cls=CallbackResult,
    )
    symbol_registry.register_builtin_registry(
        import_path="modularml.callbacks.callback_registry",
        registry=callback_registry,
        naming_fn=callback_naming_fn,
    )

    # Phases
    symbol_registry.register_builtin_class(
        key="ExperimentPhase",
        cls=ExperimentPhase,
    )
    symbol_registry.register_builtin_class(
        key="TrainPhase",
        cls=TrainPhase,
    )
    symbol_registry.register_builtin_class(
        key="EvalPhase",
        cls=EvalPhase,
    )
    symbol_registry.register_builtin_class(
        key="PhaseResults",
        cls=PhaseResults,
    )


def register_kinds():
    pass

    # kind_registry.register(
    #     cls=Experiment,
    #     kind=SerializationKind(name="Experiment", kind="exp"),
    # )
