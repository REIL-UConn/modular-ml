
<div align="center">

[![ModularML Banner](docs/_static/logos/modularml_logo_banner.png)](https://github.com/REIL-UConn/modular-ml)

**Modular, fast, and reproducible ML experimentation built for R\&D.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/modularml.svg)](https://pypi.org/project/modularml/)
[![codecov](https://codecov.io/github/REIL-UConn/modular-ml/graph/badge.svg?token=Z063M1M6P3)](https://codecov.io/github/REIL-UConn/modular-ml)
[![Docs](https://readthedocs.org/projects/modular-ml/badge/?version=latest)](https://modular-ml.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

</div>


ModularML is a flexible, backend-agnostic machine learning framework for designing, training, and evaluating machine learning pipelines, tailored specifically for research and scientific workflows.
It enables rapid experimentation with complex model architectures, supports domain-specific feature engineering, and provides full reproducibility through configuration-driven declaration.

> ModularML provides a plug-and-play ecosystem of interoperable components for data preprocessing, sampling, modeling, training, and evaluation — all wrapped in a unified experiment container.


<p align="center">
  <img src="docs/_static/figures/modularml_overview_diagram.png" alt="ModularML Overview Diagram" width="600"/>
</p>
<p align="center"><em>Figure 1. Overview of the ModularML framework, highlighting the three core abstractions: feature set preprocessing and splitting, modular model graph construction, and staged training orchestration.</em></p>



## Key Concepts and Features

### FeatureSet & FeatureSetView
- **`FeatureSet`** is the primary user-facing container for structured data. It tracks features/targets/tags, reversible transforms, and named splits.
- **`FeatureSetView`** gives a lightweight view into a FeatureSet (rows + selected columns) so you can feed exactly the slices required for a training phase.

### Splitters & Samplers
- Built-in **splitters** (e.g., random, rule-based) generate labeled splits from any FeatureSet.
- **Samplers** consume FeatureSets or views and emit `BatchView`s in the shape required by the model. They support stratification, grouping, triplets/pairs, and custom roles so you can express experiment-specific batching without re-implementing the training loop.

### Models & Wrappers
- Use your own **PyTorch or TensorFlow models**, select from pre-exiting templates, or wrap third-party estimators. ModularML provides backend wrappers (Torch, TensorFlow, scikit-learn) so any supported model exposes a consistent forward API and reports its backend.

### ModelGraph and Node-based Connectivity
- **`ModelNode`** attaches a wrapped model to an upstream FeatureSet or node, handles building, freezing, and optimizer wiring.
- **`MergeNode`** (e.g., `ConcatNode`) combines outputs from multiple nodes when you need multi-branch architectures.
- **`ModelGraph`** is the DAG that ties everything together. It resolves head/tail nodes, executes topological forward/backward passes, mixes backends, and lets you switch between stage-wise or global training with a single call.

### AppliedLoss
- **`AppliedLoss`** instances bind user-defined loss functions to nodes within the ModelGraph. They carry labels, weights, and node scopes so multi-objective training is easy to configure from a phase or experiment.

### Experiment Phases
- **`TrainPhase`** runs iterative training with your sampler schedule, losses, callbacks, and optimizer configuration.
- **`FitPhase`** (single-pass) is ideal for algorithms that expect a one-shot `.fit()` (e.g., scikit-learn estimators) after upstream neural components are frozen.
- **`EvalPhase`** executes forward passes and records losses/metrics on held-out splits without touching gradients.

### Experiment Class
- The **`Experiment`** binds FeatureSets, ModelGraph, and all phases. It owns execution order, logging, callbacks, and results objects so every run is reproducible. Execution strategies (e.g., cross validation) simply wrap an Experiment to replay the same plan across folds.

### Serialization
- A core focus of ModularML is reproducibility. To that end, all major classes (FeatureSets, ModelGraph, phases, experiments, losses, samplers, optimizers, callbacks) implement configuration/state serialization
- All model definitions, training/sampling logic, evaluation, etc is structured under a single Experiment object, allowing for exporting and sharing via a single `.mml` file.

### Callbacks & Checkpointing
- Built-in **callbacks** (EarlyStopping, Evaluation + metrics, custom progress hooks) plug directly into Train/Fit/Eval phases, allowing for fully flexibile workflows while retaining a structured experiment API.
- **Checkpointing** can be attached at any major experiment or training execution step to persist model weights, optimizer states, FeatureSet transforms, and sampler cursors, making restarts seamless.



## Getting Started

Requires Python >= 3.9

### Installation
Install from PyPI:
```bash
pip install modularml
```

To install the latest development version:
```bash
pip install git+https://github.com/REIL-UConn/modular-ml.git
```


## Explore More
- **[Explanation](docs/explanation/)** – Conceptual material that explains why ModularML is structured the way it is.
- **[How-To](docs/how_to/)** – Deep dive on core components of the ModularML framework.
- **[Tutorial](docs/tutorials/)** – Explore complete walkthroughs of solving common machine learning tasks with ModularML.
- **[Full API Reference](https://modular-ml.readthedocs.io/en/latest/)** – API reference, component explanations, configuration guides, and tutorials.
- **[Discussions](https://github.com/REIL-UConn/modular-ml/discussions)** – Join the community, ask questions, suggest features, or share use cases.

---


<!-- ## Cite ModularML

If you use ModularML in your research, please cite the following:

```bibtex
@misc{nowacki2025modularml,
  author       = {The ModularML Team},
  title        = {ModularML: Modular, fast, and reproducible ML experimentation built for R&D.
  },
  year         = {2025},
  note         = {https://github.com/REIL-UConn/modular-ml},
} -->
<!--
## The Team
ModularML was initiated in 2025 by Ben Nowacki as part of graduate research at the University of Connecticut.

The project is community-driven and welcomes contributors interested in building modular, reproducible ML workflows for science and engineering. -->

## License
**[Apache 2.0](https://github.com/REIL-UConn/modular-ml/license)**
