# ModularML Roadmap

## v1.0.0 — Core ModularML Pipeline  
![Progress](https://img.shields.io/badge/progress-67%25-yellow)

**Target Release:** Q1 2026

---

### Data Structures & Serialization
- [x] Refactor `FeatureSet`, `FeatureSubset`, `Batch`, and related structures to use **PyArrow tables**
- [x] Implement **zero-copy subset & sampler views** over parent FeatureSet tables
- [x] Ensure data loads into memory **only when needed** for ModelGraph execution
- [x] Make all components fully **serializable** (FeatureSets, ModelGraphs, Stages, Samplers, Losses, Phases)
- [x] Support exporting Experiments as:
  - [x] Full state (post-training, weights included)
  - [ ] Config-only (reproducible structure, no weights)

---

### Experiment Context & Tracking
- [x] Implement automatic **Experiment context binding** for all defined components
- [x] Add conflict detection for mismatched component/Experiment associations
- [x] Store all outputs (loss curves, metrics, results, figures) linked to their source phase

---

### FeatureSet / Splitting / Sampling

#### FeatureSet
- [x] Fully structured feature–target–tag schema
- [x] Per-column scaling/normalization with tracked transform pipelines

#### Splitting
- [x] Ratio-based random splits
- [x] Rule-based conditional splits (user-defined criteria)

#### Sampling
- [x] Sample-wise batching
- [x] N-Sampler-based paired sampling
- [x] N-Sampler-based triplet sampling

---

### ModelGraph
- [x] Support sequential, branching, and merging DAGs
- [x] Validate graph connectivity before training
- [x] Add graph visualization utility (Graphviz/Dot/Mermaid)

---

### ModelNode
- [x] Unified wrappers for PyTorch, TensorFlow, and scikit-learn
- [x] Built-in PyTorch models (Sequential MLP, CNN encoder)
- [x] Backend-agnostic forward, training-step, and eval-step APIs

---

### MergeNodes
- [x] Support merging of multi ModelGraph branches
- [x] ConcatNode for concatenating features of multiple inputs
  - [x] Add non-concat aggregation strategies for targets and tags
  - [x] Support padding of data with misaligned shapes
- [x] Make merging backend-aware to prevent PyTorch auto-grad breakage

---

### Experiment / TrainingPhase / EvaluationPhase
- [x] Experiment holds static FeatureSets, splits, and ModelGraph
- [x] Support multiple independent **Training** and **Evaluation** phases
- [x] Each phase configurable with samplers, losses, optimizers, and trackers
- [x] Store and version phase results in the Experiment instance

---
### Unit Testing
- [x] Add nox-based automated unit, integration, example, and doc test routines
- [ ] Increase code coverage to $\geq$ 90%

---

## v1.1.0 — Multi-Experiment Container & Comparison  
![Progress](https://img.shields.io/badge/progress-0%25-red)

**Target Release:** Q3 2026

- [ ] Multi-input/output Samplers
  - Samplers can take in multiple `FeatureSets`
    - Must support sample alignment (separate from `BatchSchedulingPolicy`)
  - Samplers can produce multiple output `streams`

- [ ] Add higher-level **ExperimentCollection** container
- [ ] Support grouping Experiments for shared evaluation pipelines
- [ ] Provide unified comparison utilities across Experiments (metrics, plots, tables)
- [ ] Enable rapid testing of alternative ModelGraphs, architectures, or FeatureSets within the same task
