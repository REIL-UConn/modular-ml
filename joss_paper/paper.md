---
title: 'ModularML: A Backend-Agnostic, Declarative Framework for Reproducible and Modular Machine Learning Experiments'
tags:
  - python
  - machine-learning
  - experimentation
  - reproducibility
  - research-tools
authors:
  - name: Ben Nowacki
    orcid: 0009-0000-6723-3085
    affiliation: "1"
  - name: Tingkai Li
    orcid: 0000-0003-1934-1000
    affiliation: "1"
  - name: Sina Navidi           # TODO: add orcID
    affiliation: "1"
  - name: Mohammad Mundiwala    # TODO: add orcID
    affiliation: "1"
  - name: Hui Hua               # TODO: add orcID
    affiliation: "1"
  - name: Fei Miao              # TODO: add orcID
    affiliation: "1"
  - name: Chao Hu               # TODO: add orcID
    affiliation: "1"
    corresponding: true
affiliations:
 - name: University of Connecticut, United States
   index: 1
date: 2 June 2026
bibliography: paper.bib
---

# Summary

Modern machine learning (ML), particularly deep learning, has become central to scientific research across domains such as energy systems, materials science, structural health monitoring, and biomedical engineering.
However, core ML frameworks primarily focus on model implementation and optimization, leaving experiment structure, data partitioning, and reproducibility patterns to user-defined scripts.
As a result, domain researchers often rely on fragmented scripts, backend-specific conventions, and ad hoc experiment management workflows that limit reproducibility and collaboration.

`ModularML` is an open-source Python framework designed to address this gap.
It provides a backend-agnostic, declarative, and fully serializable architecture for ML experimentation, enabling users to define complete workflows---including data curation, feature processing, model architecture, training strategy, loss configuration, and evaluation---using modular, composable components that serialize into a single reproducible artifact.

At its core, `ModularML` introduces an abstraction for structured, backend-agnostic data handling and splitting; a directed acyclic graph (DAG) of model nodes with support for shared or node-specific optimization; multi-phase training orchestration with configurable freeze/unfreeze policies and phase- and node-specific loss definitions; and full pipeline serialization and visualization for transparent sharing and inspection of experiments.
By separating experiment definition from backend implementation, `ModularML` allows researchers to collaborate and audit ML studies without requiring expertise in `PyTorch` [@pytorch], `Keras` [@keras], or other specific ML libraries.


# Statement of need

ML workflows in academia differ substantially from typical industry-focused ML development.
Scientific researchers emphasize experimentation with domain-specific feature selection, sampling, and model architecture.
Existing libraries such as `PyTorch Lightning` [@pytorch_lightning], `Catalyst` [@catalyst], `FastAI` [@fastai], and `PyTorch Ignite` [@pytorch_ignite] can streamline training loops, but typically remain tied to a single backend, prioritize development productivity over experimentation standardization, and depend on implicit code-based configurations with limited traceability of dataset splitting.

In contrast, `ModularML` is built around the principle of *configuration-as-contract*: the entire ML experiment is represented as a structured, inspectable object. This design offers three major benefits:

1. **Reproducibility through serialization.** Entire experiments, from initial data curation to multi-stage training and evaluation, can be saved and shared as a single file. A collaborator can load this artifact and inspect the full pipeline without having to trace the source code.
2. **Backend abstraction.** Model nodes are agnostic to the backend in which the underlying model is defined. `ModularML` does not replace existing ML libraries such as `PyTorch`, `Keras`, or `scikit-learn` [@scikit_learn]; rather, it complements them by providing a unified experiment structure around backend-specific model implementations. This backend-agnostic design supports reuse of collaborator models regardless of which library was used to implement them.
3. **Transparency for non-ML experts.** Built-in `summary` and `visualize` methods for feature sets, model graphs, and experiment execution phases allow users and code reviewers to quickly inspect data splits, model architecture, loss routing, and training sequencing without needing to understand backend-specific implementation details.

These features directly address common reproducibility and collaboration challenges in research, while still supporting a comprehensive suite of modeling techniques and existing ML libraries.
In particular, `ModularML` is intended to serve as an experiment-structure layer rather than a replacement for backend frameworks: users can continue relying on established libraries for model implementation and numerical execution while using `ModularML` to define, serialize, and inspect the full experimental workflow.
This makes the framework especially useful in collaborative research settings where model design, data partitioning, and training logic must be reviewed across users with different levels of ML software expertise.

# State of the field

General-purpose deep learning frameworks (e.g., `PyTorch`, `Keras`) streamline model training but typically assume homogeneous backends and single-phase workflows.
Data-centric libraries (e.g., `Apache Arrow` [@pyarrow], `Hugging Face` [@huggingface_datasets]) manage datasets but leave sampling and experiment scheduling to custom user code.
Experiment managers (e.g., `MLflow` [@mlflow], `Weights & Biases` [@wandb]) track runs but do not define how data flows through modular graphs.
`ModularML` sits between these layers.
It provides concrete abstractions for data (`modularml.FeatureSet`), sampling logic (`modularml.Sampler`), modeling (`modularml.ModelGraph` composed of `modularml.ModelNode`s), losses (`modularml.AppliedLoss`), and orchestration (`modularml.ExperimentPhase`), while remaining backend-agnostic and lightweight enough for research scripts or notebooks.


# Software design

The `ModularML` architecture provides abstractions for data storage and processing, graph-based modeling, and execution of distinct training sequences (\autoref{fig:architecture}).
Rather than embedding these responsibilities within custom training scripts, `ModularML` separates them into explicit, composable components that can be independently configured, inspected, and serialized.
This separation enables rapid experimentation with data pipelines, model architectures, and training strategies while preserving reproducibility and traceability across the entire ML workflow.

![ModularML architecture overview: FeatureSet curation into samples with feature, target, and tag domains; flexible ModelGraph construction for rapid experimentation of model topology; and multi-phase training workflows sequenced within a single Experiment container.\label{fig:architecture}](modularml_overview_diagram.png)

## FeatureSets

A `modularml.FeatureSet` organizes data into three intent-driven domains---features, targets, and tags---reflecting the core design philosophy of the data abstraction.
Features represent model inputs (what the model learns from), targets represent model outputs (what the model is trained to predict or reproduce), and tags store optional metadata associated with each sample.
Each sample, defined by its feature, target, and tag attributes, is assigned a globally unique identifier to ensure explicit traceability throughout the entire lifecycle of an experiment.
This identifier propagates through splitting, sampling, batching, model execution, and evaluation, enabling transparent lineage tracking and reproducible analysis.

Data within a `modularml.FeatureSet` is stored in backend-agnostic containers supporting `NumPy` [@numpy] arrays, `Pandas` [@pandas] dataframes, `PyTorch` tensors, and `TensorFlow` [@tensorflow] tensors.
All downstream operations, such as splitting, subsetting, and batching, operate using no-copy views of the underlying data.
This design makes split definitions explicit and inspectable, reduces memory overhead, and minimizes the risk of data leakage or unintended experimentation bias.

## Sampling
All subclasses of `modularml.Sampler` consume a `modularml.FeatureSet`, or subset views of one, and emit aligned batches, supporting stratification, grouping, and multi-role sampling needed for contrastive or paired training schemes.
By separating sampling logic from model execution, `ModularML` makes data selection strategies explicit and reproducible rather than embedding them within custom training loops.
This design allows researchers to experiment with different batching and sampling approaches without modifying model or training code.

## ModelGraph
The full experiment model is represented as a DAG of interconnected `modularml.ModelNode`s.
Each node wraps a user-defined or built-in ML model and exposes standardized interfaces for construction, optimization, and freezing.
Specialized node types, such as `modularml.MergeNode`, extend this functionality by supporting multiple inputs and configurable merge operations including concatenation, aggregation, and padding.
Together, these nodes form a `modularml.ModelGraph`, which manages dependency resolution and executes forward and backward passes in topological order.
This graph-based design separates model topology from training logic, enabling rapid experimentation with branching architectures, multi-stage pipelines, and hybrid systems that combine learned and classical ML components.

## Losses and phases
`modularml.AppliedLoss` binds objectives to specific nodes within a `modularml.ModelGraph`, enabling composite-loss training and targeted loss aggregation across different portions of a model.
These losses are tracked throughout execution, providing detailed records for post-training analysis and comparison of experimental configurations.

Model execution is organized into a sequence of `modularml.ExperimentPhase` objects.
Each phase defines a reproducible unit of execution, including data sampling, loss definitions, callbacks, and training or evaluation behavior.
Built-in phase types include iterative training (`modularml.TrainPhase`), single-pass fitting workflows (`modularml.FitPhase`), and inference-only evaluation (`modularml.EvalPhase`), providing a consistent interface for constructing complex multi-phase ML workflows.

## Experiment orchestration
The `modularml.Experiment` class serves as the top-level container for a ML workflow, binding together one or more `modularml.FeatureSet`s, a `modularml.ModelGraph`, and a sequence of execution phases.
By explicitly separating experiment definition from execution, `ModularML` supports workflows such as pretraining, fine-tuning, and evaluation while retaining all information required to reproduce each stage.
The `modularml.Experiment` also provides serialization, checkpointing, and execution tracking, allowing complete experiments to be exported, shared, and reloaded as self-contained artifacts.


# Research impact statement

`ModularML` provides structured abstractions for scientists prototyping hybrid ML systems that combine learned encoders, classical regressors/classifiers, and domain-specific samplers.
By guaranteeing that every component can be serialized, checkpointed, and replayed, the framework supports reproducible experiments and facilitates sharing of trained graphs or datasets between collaborators.
Its backend-agnostic graph execution simplifies comparisons across `PyTorch`, `TensorFlow`, and `scikit-learn` implementations, encouraging rigorous benchmarking and cross-validation in applied research domains.

The unique contributions can be summarized as follows:

1. **Full pipeline serialization.** Rather than serializing only model weights, `ModularML` serializes data definitions with assigned unique identifiers, split logic, sampling configuration, graph topology, per-phase loss routing, and training and evaluation configurations. This enables complete auditability of published ML studies, as shown in \autoref{fig:collaboration}.
2. **Backend-agnostic DAG modeling.** By supporting backend-agnostic modeling, `ModularML` extends adoption of existing ML libraries rather than replacing them, reducing friction between works that use different packages.
3. **Declarative syntax.** All aspects of an ML experiment are constructed via declarative, configuration-driven objects. This makes experiment definitions explicit, simplifies review, and supports systematic iteration across training techniques and model architectures.
4. **Built-in visualization for validation.** Visual summaries help detect data leakage, misconfigured sampling, incorrect loss routing, unexpected model topologies, and training sequencing issues (\autoref{fig:collaboration}). This enables validation at definition time, ensures execution matches intent, and lowers the barrier for domain scientists to review ML-based results.

![Backend-agnostic and fully serializable experiment workflow in ModularML with built-in visualization utilities.\label{fig:collaboration}](modularml_collaboration.png)

# Mathematics

`ModularML` does not introduce new mathematical formulations; instead, it codifies well-established training loops, loss aggregation, and optimizer steps across common ML backends.
Any model- or loss-specific mathematics is delegated to user-defined modules or external libraries, ensuring that `ModularML` remains an orchestration layer that provides continued support for existing ML packages.

# AI usage disclosure

Generative AI tools were used in a limited capacity during the development of this software, confined to helping structure docstrings and scaffold unit tests.
All AI-assisted contributions were reviewed and verified by the authors.
AI was not used in the writing of this manuscript.

# Acknowledgements

We thank the `ModularML` contributor community for feature ideas, bug reports, and documentation improvements, and acknowledge Professor Ryan Cooper for guidance in navigating the open-source environment.

# References
