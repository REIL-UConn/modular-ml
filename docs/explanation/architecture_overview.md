# Architecture Overview

ModularML is a framework for building modular, composable machine learning pipelines. Rather than treating a model as a monolithic block that consumes raw data and produces predictions, ModularML decomposes the workflow into distinct, interchangeable layers: **data management**, **graph topology**, **training orchestration**, and **serialization**. Each layer has a clear responsibility and communicates with the others through well-defined interfaces.

This document provides a bird's-eye view of those layers and the reasoning behind their design. For deeper discussion of individual topics, see [Model Graph Design](model_graph_design.md), [Training Phases](training_phases.md), and [Experiment Desgin](experiment_design.md).

## Why layers?

Many ML frameworks conflate data handling, model definition, and training logic into a single workflow. This works well for simple cases, but becomes unwieldy when experiments grow in complexity: multiple input streams, multi-stage models, ensemble strategies, or cross-validation folds that each require different data partitions.

ModularML addresses this by separating concerns into layers that can be composed independently. A `FeatureSet` does not know what model will consume it. A `ModelGraph` does not know how its nodes will be trained. An `Experiment` does not know the internal structure of the graph it orchestrates. This separation means that changing one part of the pipeline, say swapping a sampler or adding a new model stage, does not require rewriting the rest.

## The four layers

The framework is organized into four conceptual layers, each building on the one below it.

### Layer 1: Data storage

```{mermaid}
---
title: Layer 1 - Data Storage
---
graph TB
    FeatureSetView -->|"view into"| FeatureSet
    FeatureSet --> SampleCollection
    SampleCollection --> PyArrow["PyArrow Table"]

```

At the foundation sits the `FeatureSet`, ModularML's central data container. Internally, a FeatureSet holds a `SampleCollection`—an immutable Apache Arrow table whose columns follow a structured naming convention: `<domain>.<key>.<representation>`. For example, `features.velocity.raw` or `targets.label.transformed`.

This naming scheme is deliberate. The **domain** (features, targets, tags, sample_uuids) tells the framework what role a column plays. The **key** is the user-defined name. The **representation** tracks whether data is in its original form (`raw`) or has been processed by a scaler (`transformed`). When a scaler is applied, the original data is never overwritten; instead, a new `transformed` column appears alongside the `raw` one. This makes it straightforward to undo transforms or inspect what the scaler actually changed.

Every sample is also assigned a UUID, which persists through splits and transforms. This means a sample can be traced from the original dataset all the way through training, regardless of how many views, batches, or folds it passes through.

The choice of Apache Arrow as the storage backend reflects a preference for columnar, zero-copy data access. Arrow tables can be sliced without copying memory, which is what enables `FeatureSetView`—a lightweight object that holds only a set of row indices and column names pointing back into the parent FeatureSet. Splits, column selections, and subset operations all produce views rather than copies.

### Layer 2: Data processing

```{mermaid}
---
title: Layer 2 - Data Processing
---
graph LR
    Sampler -->|"draws from"| FeatureSetView
    Sampler -->|"produces"| Batch
    Splitter <--> FeatureSetView
    Scaler <-->|"representations"| SampleCollection
```

Between raw storage and model consumption sit three processing components: **splitters**, **scalers**, and **samplers**. Each transforms or partitions data in a specific way, and each is tracked so its effects can be inspected or undone.

**Splitters** partition a FeatureSet into named subsets (e.g., train, validation, test). The result is a dictionary of `FeatureSetView` objects, each pointing to a disjoint set of rows in the original data. Because views are zero-copy, splitting a million-row dataset is inexpensive. The FeatureSet records which splitters have been applied via `SplitterRecord` objects, providing a reproducible audit trail.

**Scalers** wrap transformation logic (normalization, standardization, custom transforms) behind a uniform interface. A scaler can be specified by name (`"minmax"`), by class, or by instance. Regardless of whether a built-in or *scikit-learn* scaler is used, the  internal `Scaler` wrapper standardizes the fitting and transformation process while providing robust serialization. When applied via `fit_transform()`, the scaler learns its parameters from the data and writes the transformed values to the `'transformed'` representation columns. The FeatureSet records each applied scaler using `ScalerRecord` objects, preserving the exact application order. This enables `undo_last_transform()` and `undo_all_transforms()` to restore previous states, which is especially useful when experimenting with alternative preprocessing strategies.

**Samplers** convert FeatureSet data into batches suitable for model consumption. A sampler binds to one or more FeatureSetView objects and yields `Batch` instances through Python’s iterator protocol. Each Batch represents a single unit of model execution and contains all inputs required for that step. Samplers utilize two string-based terms when generating batches: `roles` and `streams`.

* **Roles** define the concurrent inputs that must be sampled and processed together within a batch. Each role corresponds to a named input context, such as `"anchor"`, `"positive"`, and `"negative"` in metric learning, or `"anchor"` and `"pair"` in contrastive learning. Roles ensure that related samples are aligned and available simultaneously so the model and loss functions can operate on their relationships.
* **Streams** define optional named output branches produced by a sampler. Most workflows use a single stream, but streams allow advanced samplers to emit multiple structured outputs with explicit naming and routing. This enables more complex execution patterns, such as branching inputs to different model components or training objectives.

Internally, batch contents are stored as `RoleData`, which maps role names to `SampleData` objects. Each SampleData encapsulates the domain-structured tensors associated with that role, including features, targets, tags, and UUIDs. This structure provides explicit separation between different input roles while preserving consistent access patterns.

Although the role and stream abstractions add structure, they remain lightweight in simple cases. When only a single role and stream are used, specifiers can be ommitted entirely, allowing the batch object to behave like a standard feature-target container. However, this design scales naturally to more complex scenarios involving multiple coordinated inputs or distinct preprocessing paths, without requiring changes to the surrounding training pipeline.


### Layer 3: Graph topology

```{mermaid}
---
title: Layer 3 - Graph Topology
---
graph TB
    ModelGraph --> ModelNode
    ModelGraph --> MergeNode
    MergeNode --> ConcatNode
    ModelNode --> BaseModel
```

ModularML represents model architectures as a directed acyclic graph (DAG) of nodes. The base class, `GraphNode`, provides identity, labeling, and upstream/downstream wiring. Its subclass `ComputeNode` adds a `forward()` method, and the two concrete compute node types are `ModelNode` (wrapping a trainable or static model) and `MergeNode` (combining outputs from multiple upstream nodes).

A `ModelNode` holds a `BaseModel` instance—an abstraction over backend-specific implementations. `BaseModel` has subclasses for *PyTorch* (`TorchBaseModel`), *TensorFlow* (`TensorflowBaseModel`), and *scikit-learn* (`ScikitWrapper`). This means the graph topology is defined independently of the ML backend. A graph can mix nodes from different backends, though in practice most workflows use a single backend throughout.

`MergeNode` and its subclass `ConcatNode` handle cases where multiple upstream outputs need to be combined before feeding into a downstream node. This is common in multi-input architectures, ensemble designs, or feature fusion strategies.

Nodes are wired together through **references**—symbolic pointers that are resolved at execution time rather than construction time. A `FeatureSetReference` points to specific columns of a FeatureSet, while a `ModelIOReference` points to a node's input or output. This lazy resolution means that a graph can be defined before the data it will consume exists, which is important for serialization, cross-validation (where the same graph structure is applied to different data folds), and experiment templating.

The `ModelGraph` itself is the container that owns all nodes and manages execution. Its `forward()` method performs a topological traversal, passing each node's output as input to its downstream neighbors. The `build()` method triggers shape inference, propagating tensor dimensions through the graph so that shape mismatches are caught before training begins.

For a deeper discussion of graph composition patterns and design principles, see [Model Graph Design](model_graph_design.md).

### Layer 4: Orchestration

```{mermaid}
---
title: Layer 4 - Orchestration
---
graph TB
    Experiment --> PhaseGroup
    PhaseGroup --> TrainPhase
    PhaseGroup --> EvalPhase
    PhaseGroup --> FitPhase
```

The orchestration layer ties everything together through three components: the `ExperimentContext`, `Experiment`, and **phases**.

The `ExperimentContext` is a registry that tracks all nodes (FeatureSets, ModelGraphs, ModelNodes) by ID and label. It acts as a namespace that references are resolved against. The context uses a thread-local singleton pattern (`ContextVar`), which means each experiment operates in isolation even in concurrent settings.

An `Experiment` is the top-level orchestrator. It holds a sequence of phases (organized into a `PhaseGroup`), manages checkpointing, and records execution history. Calling `experiment.run()` iterates through all registered phases in order, tracking all required results and performing any attached `Callbacks` at any transition point (e.g., at the start and end of phases, epochs, batches, etc.).

**Phases** define what happens during execution. A `TrainPhase` iterates through a sampler's batches, runs each batch through the model graph's forward pass, computes a loss, performs backpropagation, and steps the optimizer. An `EvalPhase` does the same but without gradient computation. Both support callbacks for logging, early stopping, metric computation, and other side effects. A separate `FitPhase` is introduced for closed-form models (e.g., *scikit-learn* regressors) where all training data is fit to in a single pass, rather than the iterative optimization used in TrainPhase.

The `AppliedLoss` object binds a loss function to a specific node's output and a target domain, telling the phase exactly which predictions to compare against which targets. This indirection is necessary because a graph can have multiple output nodes, each with its own loss. Mirroring the pattern of the scalers, AppliedLoss wraps any backend-specific or custom loss function with the `Loss` class. This is how we enable full serialization of losses, regardless of whether they are a built-in method or some user-created callable.

During each batch pass, an `ExecutionContext` is created to hold the transient state: which inputs were fed to which head nodes, what each node produced, and what losses were accumulated. This context is available to callbacks and loss functions, giving them full visibility into the current execution state without requiring global mutables.

For more on how phases structure the training lifecycle, see [Training Phases](training_phases.md).

## Cross-cutting concerns

Several design decisions cut across all four layers.

### Serialization

Every major component implements the `Configurable` protocol (`get_config()`) and most also implement `Stateful` (`get_state()` / `set_state()`). The distinction is intentional: configuration captures *structure* (what type of scaler, what graph topology, what loss function), while state captures *runtime values* (learned scaler parameters, model weights, optimizer momentum). This means an experiment's structure can be saved and shared independently of its trained state, which is useful for experiment templating and reproducibility.

### Registries

Extensible components—samplers, splitters, scalers, losses, optimizers, models—are registered in named registries. This allows them to be referenced by string identifier (e.g., `"minmax"`, `"mse"`) rather than by direct import, which simplifies serialization and configuration files. User-defined components can be added to the same registries, making them first-class citizens alongside built-in implementations.

### Backend neutrality

ModularML maintains explicit backend awareness using a `Backend` enum (TORCH, TENSORFLOW, NUMPY) to tag data and models with their execution backend. Conversion utilities handle translation between backends when required, and core data containers such as Batch and SampleData can automatically convert their contents to match the backend expected by a model.
This design prioritizes interoperability while acknowledging that backend differences are fundamental. Rather than hiding these differences, each backend-specific model wrapper is responsible for implementing its own forward and backward execution logic.

Explicit backend tracking also enables pre-execution validation of backend compatibility across a ModelGraph. This allows the framework to detect when backend conversions would break gradient propagation and determine whether graph-wide training is possible (when all stages share a compatible backend) or whether training must be performed independently at each stage.

### Zero-copy data access

The combination of Arrow-backed storage and index-based views means that data is rarely copied during normal workflow operations. Splitting a FeatureSet, selecting columns, creating batches—all produce lightweight views or slices rather than full copies. Actual data materialization happens at the boundary where tensors are handed to model frameworks, which is typically the latest possible moment. This minimizes the memory overhead during Experiment structuring, offloading any additional materialization memory costs to runtime.

## How the pieces fit together

A typical ModularML workflow moves through the layers in sequence:

1. **Data ingestion**: Create a `FeatureSet` from a dictionary, DataFrame, or Arrow table. Apply splitters to create train/validation/test views. Apply scalers to normalize features.

2. **Graph construction**: Define `ModelNode` instances, each wrapping a backend-specific model. Wire them together with references to FeatureSets and to each other. Wrap everything in a `ModelGraph`.

3. **Experiment setup**: Create an `Experiment` with a sequence of phases. Each phase specifies which graph node to execute, which sampler to use, and (for training) which loss(es) to attach and which nodes to optimize.

4. **Execution**: Call `experiment.run()`. The framework iterates through phases, which iterate through batches, which flow through the graph, producing outputs and computing losses.

5. **Persistence**: Save checkpoints, export model state, or serialize the full experiment for reproduction.

Each step is independent enough that it can be modified without affecting the others. Swapping a sampler does not require changing the graph. Adding a new model stage does not require rewriting the experiment. This composability is the central design goal of ModularML.
