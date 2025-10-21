# TODO – Refactor Roadmap
1. **Experiment Orchestration Layer**
   - Replace hard-coded repo/model loops in `evaluator.py` with an `ExperimentRunner` that reads structured configs (YAML/JSON) covering model × repo matrices, repetition counts, and shared environment settings.
   - Keep `ConfigurationEvaluator` stateless; have the runner schedule runs, persist aggregated metrics, and prepare thesis-friendly CSV/JSON outputs.
2. **Repository Lifecycle Management**
   - Refactor `RepositoryManager` into a context-managed `RepositoryWorkspace` that allocates unique workdirs per run, exposes typed file APIs, and guarantees cleanup.
   - Remove mutable global state (`_tmp_dir`, `_repo_name`) and stringly-typed responses to support concurrent experiments safely.
3. **Validation Pipeline Modularization**
   - Decompose `ConfigurationValidator` into pluggable `ValidationStep`s (syntax, Hadolint, build, apply, runtime) coordinated by a pipeline runner.
   - Inject command runners to mock external binaries in tests and record per-step timings, severities, and tool availability for reporting.
4. **Artifact Store for Generated Outputs**
   - Introduce an `ArtifactStore` that captures generated Dockerfiles/K8s manifests into run-scoped directories, tracks provenance, and exposes them for reports.
   - Update generator/validator collaboration to reference artifacts via the store instead of raw repo paths.
5. **Layered Configuration Models**
   - Split `GeneratorConfig` into `EnvConfig`, `ModelConfig`, and `RunConfig` so environment secrets, LLM parameters, and per-experiment toggles remain isolated.
   - Provide serialization/loading helpers to swap models or temperatures without mutating global state.
6. **Unified Metrics & Reporting**
   - Define a `RunMetrics` aggregate capturing phase durations, tool usage, scores, and external command results.
   - Have `EvaluationReporter` consume this structure to emit JSON/CSV/HTML consistently and simplify thesis analysis.
7. **Domain-Specific Completeness Checks**
   - Implement a `CompletenessValidator` module that detects missing Kubernetes resources, mismatched selectors, ConfigMap/Secret references, PVC dependencies, and port inconsistencies across Dockerfiles, Deployments, Services, and Ingresses.
   - Produce completeness/consistency scores derived from detected issues and surface them via the metrics pipeline.
8. **Reference Comparison Toolkit**
   - Add a `ComparisonValidator` able to compare generated manifests against curated “golden” configs using semantic YAML diffing (e.g., DeepDiff) and field-level checks for deployments, services, and ingress resources.
   - Persist structural similarity metrics (resource coverage, field coverage) alongside evaluation reports for ground-truth benchmarking.
9. **LLM Behaviour Analytics**
   - Create an `LLMMetricsAnalyzer` that inspects LangSmith traces for tool efficiency, redundant calls, context usage, reasoning quality, and token efficiency per generated artifact.
   - Extend the experiment runner to schedule repeated runs (same seed/model) and compute consistency metrics such as resource variance, value stability, and quality score dispersion.
