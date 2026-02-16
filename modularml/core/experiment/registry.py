from modularml.core.experiment.experiment import Experiment
from modularml.core.experiment.phases.phase_group import PhaseGroup
from modularml.core.experiment.results.eval_results import EvalResults
from modularml.core.experiment.results.group_results import PhaseGroupResults
from modularml.core.experiment.results.train_results import TrainResults
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from modularml.core.experiment.callback import Callback, CallbackResult
from modularml.callbacks import callback_naming_fn, callback_registry

from modularml.core.experiment.phases.phase import ExperimentPhase
from modularml.core.experiment.phases.train_phase import TrainPhase
from modularml.core.experiment.phases.eval_phase import EvalPhase
from modularml.core.experiment.results.phase_results import PhaseResults


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
        key="PhaseGroup",
        cls=PhaseGroup,
    )
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

    # Result containers
    symbol_registry.register_builtin_class(
        key="PhaseGroupResults",
        cls=PhaseGroupResults,
    )
    symbol_registry.register_builtin_class(
        key="PhaseResults",
        cls=PhaseResults,
    )
    symbol_registry.register_builtin_class(
        key="EvalResults",
        cls=EvalResults,
    )
    symbol_registry.register_builtin_class(
        key="TrainResults",
        cls=TrainResults,
    )

    # Experiment
    symbol_registry.register_builtin_class(
        key="Experiment",
        cls=Experiment,
    )


def register_kinds():
    kind_registry.register(
        cls=Experiment,
        kind=SerializationKind(name="Experiment", kind="exp"),
    )
