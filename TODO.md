# TODO – Refactor Roadmap
1. ✅ **Experiment Orchestration Layer** – DONE
   - ✅ Replace hard-coded repo/model loops in `evaluator.py` with an `ExperimentRunner` that reads structured configs (YAML/JSON) covering model × repo matrices, repetition counts, and shared environment settings.
   - ✅ Keep `ConfigurationEvaluator` stateless; have the runner schedule runs, persist aggregated metrics, and prepare thesis-friendly CSV/JSON outputs.
   - ✅ Core runner, CLI entry point, report metadata, and starter configs are implemented.
   - ✅ Experiment configs now accept prompt variants (`prompts:`) referencing alternate system-prompt files so runs can compare instruction sets.
   - **Follow-up work**: Aggregated statistics (mean/variance per model–repo pair), automating prompt metadata analysis in summaries, and hooks for future scoring/analytics integration.
2. **Parallel Experiment Execution**
   - Implement parallel execution support for running multiple experiments concurrently.
   - Handle resource conflicts: namespace collisions, URL conflicts, Docker image name conflicts, etc.
   - Ensure workspace isolation between parallel runs to prevent interference.
   - Consider implementing run-specific prefixes/suffixes for namespaces, images, and other shared resources.
   - Initial focus should remain on single-threaded execution to ensure deterministic behavior and stability.
3. **Repository Lifecycle Management**
   - Refactor `RepositoryManager` into a context-managed `RepositoryWorkspace` that allocates unique workdirs per run, exposes typed file APIs, and guarantees cleanup.
   - Remove mutable global state (`_tmp_dir`, `_repo_name`) and stringly-typed responses to support concurrent experiments safely.
4. **Validation Pipeline Modularization**
   - Decompose `ConfigurationValidator` into pluggable `ValidationStep`s (syntax, Hadolint, build, apply, runtime) coordinated by a pipeline runner.
   - Route external tool invocations through a command runner abstraction to capture timings, severities, and tool availability for reporting.
5. **Artifact Store for Generated Outputs**
   - Introduce an `ArtifactStore` that captures generated Dockerfiles/K8s manifests into run-scoped directories, tracks provenance, and exposes them for reports.
   - Update generator/validator collaboration to reference artifacts via the store instead of raw repo paths.
   - **Save generated configurations on each run**: Persist the complete generated configuration (including all manifests, Dockerfiles, and intermediate outputs) for each experiment run to enable retrospective analysis, debugging, and comparison across runs without needing to regenerate.
6. **Layered Configuration Models**
   - Split `GeneratorConfig` into `EnvConfig`, `ModelConfig`, and `RunConfig` so environment secrets, LLM parameters, and per-experiment toggles remain isolated.
   - Provide serialization/loading helpers to swap models or temperatures without mutating global state.
7. **Unified Metrics & Reporting**
   - Define a `RunMetrics` aggregate capturing phase durations, tool usage, scores, and external command results.
   - Have `EvaluationReporter` consume this structure to emit JSON/CSV/HTML consistently and simplify thesis analysis.
   - Explore deeper analysis of prompt-driven deltas in summaries (averages by `prompt_id`, highlighting metric shifts between prompt files).
8. **Domain-Specific Completeness Checks**
   - Implement a `CompletenessValidator` module that detects missing Kubernetes resources, mismatched selectors, ConfigMap/Secret references, PVC dependencies, and port inconsistencies across Dockerfiles, Deployments, Services, and Ingresses.
   - Produce completeness/consistency scores derived from detected issues and surface them via the metrics pipeline.
9. **Reference Comparison Toolkit**
   - Add a `ComparisonValidator` able to compare generated manifests against curated "golden" configs using semantic YAML diffing (e.g., DeepDiff) and field-level checks for deployments, services, and ingress resources.
   - Persist structural similarity metrics (resource coverage, field coverage) alongside evaluation reports for ground-truth benchmarking.
10. **LLM Behaviour Analytics**
    - Create an `LLMMetricsAnalyzer` that inspects LangSmith traces for tool efficiency, redundant calls, context usage, reasoning quality, and token efficiency per generated artifact.
    - Extend the experiment runner to schedule repeated runs (same seed/model) and compute consistency metrics such as resource variance, value stability, and quality score dispersion.
    - Add success-rate tracking (e.g., working configuration count out of N runs per repo/model/parameter combo) and expose it in experiment summaries for quick consistency insights.
11. **LangSmith Evaluations Integration**
    - Investigate integrating LangSmith evaluation runs to score generated artifacts with automated rubric/checklist evaluators and capture qualitative feedback alongside existing metrics.
    - Teach the experiment runner to optionally trigger LangSmith evaluations per run and persist the resulting scores in experiment summaries for cross-model analysis.
12. **Scoring Model Overhaul**
    - Replace the placeholder penalty-based scoring in `ConfigurationEvaluator` with a rubric that weights completeness, correctness, best practices, and runtime success explicitly.
    - Align the new scoring strategy with completeness/comparison validators and potential LangSmith evaluations so metrics remain comparable across experiments.
    - **Consider LLM-as-a-Judge**: Explore using an LLM (e.g., GPT-4, Claude) as an additional validation step to assess generated configurations against criteria like best practices, security, maintainability, and architectural soundness. This could complement automated validators with qualitative assessments and provide richer feedback for model comparison.
13. **Experiment Dashboard/UI**
   - Design a lightweight UI (web or TUI) to select experiment configs, kick off runs, and stream progress/status updates in real time.
   - Surface experiment summaries (per repo/model/prompt) with filters and quick access to report artifacts for faster analysis.
   - Score summary per prompt now renders in the dashboard; consider adding complementary groupings (per model, per repo, etc.) for richer comparisons.
   - Display live progress in the dashboard, including completed vs. remaining runs for the active experiment and an estimated time to completion (minutes).
14. **Agent Tooling Review**
   - Audit which tools the LLM agent should access by default and document the chosen set.
   - Decide whether adding a dedicated search capability is worthwhile and if `tree` offers benefits beyond existing `ls` support.
15. **Repository Dataset Builder**
   - Create a module that discovers and curates a dataset of GitHub repositories suited for evaluator benchmarking.
   - Support configurable filters (language, stars, topics) and persist metadata so experiments can sample consistent repo sets.
