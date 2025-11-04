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
3. ✅ **Repository Lifecycle Management** – DONE
   - ✅ Refactor `RepositoryManager` into a context-managed `RepositoryWorkspace` that allocates unique workdirs per run, exposes typed file APIs, and guarantees cleanup.
   - ✅ Remove mutable global state (`_tmp_dir`, `_repo_name`) and stringly-typed responses to support concurrent experiments safely.
4. ✅ **Validation Pipeline Modularization** – DONE
   - ✅ Decompose `ConfigurationValidator` into pluggable `ValidationStep`s (syntax, Hadolint, build, apply, runtime) coordinated by a pipeline runner.
   - ✅ Route external tool invocations through a command runner abstraction to capture timings, severities, and tool availability for reporting.
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
8. **Validator Issue Aggregation Model** ✅ COMPLETED
   - ✅ Designed and implemented a comprehensive scoring model that aggregates validator errors, warnings, and severities
   - ✅ Phase-based scoring with different weights (syntax: 40%, build: 40%, linters: 20% for Docker)
   - ✅ Severity-based penalties (ERROR: -15pts, WARNING: -5pts, INFO: -1pt)
   - ✅ Component-level aggregation (Docker: 35%, K8s: 40%, Runtime: 25%)
   - ✅ Reproducible scoring with deterministic weights
   - ✅ Per-phase breakdowns for debugging via `scoring_breakdown` field
   - ✅ Backward compatible with existing `dockerfile_score` and `k8s_manifests_score`
   - ✅ Comprehensive test suite (test_scoring_model.py)
   - ✅ Documentation (docs/scoring_model.md)
9. **Domain-Specific Completeness Checks**
   - Implement a `CompletenessValidator` module that detects missing Kubernetes resources, mismatched selectors, ConfigMap/Secret references, PVC dependencies, and port inconsistencies across Dockerfiles, Deployments, Services, and Ingresses.
   - Produce completeness/consistency scores derived from detected issues and surface them via the metrics pipeline.
10. **Reference Comparison Toolkit**
   - Add a `ComparisonValidator` able to compare generated manifests against curated "golden" configs using semantic YAML diffing (e.g., DeepDiff) and field-level checks for deployments, services, and ingress resources.
   - Persist structural similarity metrics (resource coverage, field coverage) alongside evaluation reports for ground-truth benchmarking.
11. **LLM Behaviour Analytics**
    - Create an `LLMMetricsAnalyzer` that inspects LangSmith traces for tool efficiency, redundant calls, context usage, reasoning quality, and token efficiency per generated artifact.
    - Extend the experiment runner to schedule repeated runs (same seed/model) and compute consistency metrics such as resource variance, value stability, and quality score dispersion.
    - Add success-rate tracking (e.g., working configuration count out of N runs per repo/model/parameter combo) and expose it in experiment summaries for quick consistency insights.
12. **LangSmith Evaluations Integration**
    - Investigate integrating LangSmith evaluation runs to score generated artifacts with automated rubric/checklist evaluators and capture qualitative feedback alongside existing metrics.
    - Teach the experiment runner to optionally trigger LangSmith evaluations per run and persist the resulting scores in experiment summaries for cross-model analysis.
13. **Scoring Model Overhaul**
    - Replace the placeholder penalty-based scoring in `ConfigurationEvaluator` with a rubric that weights completeness, correctness, best practices, and runtime success explicitly.
    - Align the new scoring strategy with completeness/comparison validators and potential LangSmith evaluations so metrics remain comparable across experiments.
    - **Consider LLM-as-a-Judge**: Explore using an LLM (e.g., GPT-4, Claude) as an additional validation step to assess generated configurations against criteria like best practices, security, maintainability, and architectural soundness. This could complement automated validators with qualitative assessments and provide richer feedback for model comparison.
14. ✅ **Experiment Dashboard/UI** – DONE
    - ✅ Design a lightweight UI (web or TUI) to select experiment configs, kick off runs, and stream progress/status updates in real time.
    - ✅ Surface experiment summaries (per repo/model/prompt) with filters and quick access to report artifacts for faster analysis.
    - ✅ Score summary per prompt now renders in the dashboard; consider adding complementary groupings (per model, per repo, etc.) for richer comparisons.
    - ✅ Display live progress in the dashboard, including completed vs. remaining runs for the active experiment and an estimated time to completion (minutes).
    - **Follow-up work**: Visualize run completion trends, highlight failures inline, support comparative views (model × prompt) with pinned baselines, and layer in manifest/Dockerfile diffing for side-by-side artifact analysis.
15. ✅ **Agent Tooling Review** – DONE
    - ✅ Audit which tools the LLM agent should access by default (include base64 encode/decode capability) and document the chosen set.
    - ✅ Evaluate gaps like repository search, tree listings, and data transformations; deprecate redundant tools to keep the surface area lean.
16. **Repository Dataset Builder**
    - Create a module that discovers and curates a dataset of GitHub repositories suited for evaluator benchmarking.
    - Support configurable filters (language, stars, topics) and persist metadata so experiments can sample consistent repo sets.
17. **Namespace Manifest Guardrails**
    - Update validation/generation flows to ignore or reject Kubernetes manifests that declare a `Namespace`, preventing illegal namespace creation in evaluated outputs.
18. **Masters Thesis Experiment Prep**
    - Draft the experiment design for the thesis, selecting representative repos, model variants, and prompt baselines.
    - Write and vet the key hypotheses that the thesis experiments must validate.
19. **Experiment Insights Dashboard**
    - Build a thesis-friendly dashboard that visualizes experiment outcomes with comparative charts (per model, repo, prompt).
    - Automate data analysis over summary CSV/JSON outputs to surface trends, statistical aggregates, and notable regressions.
    - Prepare reusable plots (time series, distribution, score heatmaps) for inclusion in the thesis and future presentations.
20. ✅ **Runtime Toolkit Extraction** – DONE
    - ✅ Package the deployment-critical pieces originally baked into evaluator steps into a reusable toolkit under `src/runtime/`, while keeping linting/validation concerns separate.
    - ✅ Decouple Docker builds so `DockerImageBuilder` accepts plain `DockerImageInfo` inputs, relies on `CommandRunner`, and returns structured metrics for downstream schedulers.
    - ✅ Extract Kubernetes orchestration helpers (namespace lifecycle, manifest patching, ingress discovery) and expose them via `KubernetesDeployer`/`IngressRuntimeChecker`.
    - **Follow-up work**: Add targeted unit tests for the new runtime helpers and surface a thin facade that the future PaaS API can import without touching evaluator internals.
21. ✅ **LLM Provider Expansion** – DONE
    - ✅ **Automatic OpenRouter Integration**: Implemented intelligent provider detection that automatically routes unknown providers through OpenRouter's OpenAI-compatible API, eliminating the need to install/upgrade individual LangChain provider packages for each new model.
    - ✅ **Universal Model Access**: System now supports any model available on OpenRouter (DeepSeek, Qwen, Zhipu/GLM, Mistral, Meta Llama 4, and future models) without code changes—just specify provider and model name in experiment YAML.
    - ✅ **Simplified Configuration**: Environment setup requires only `OPENROUTER_API_KEY` environment variable for all non-native providers, following LangChain's standard API key management pattern (like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
   - ✅ **Smart Model Name Formatting**: Automatically formats model names as `provider/model-name` for OpenRouter (e.g., `meta-llama/llama-4-scout`) based on provider and name fields in experiment configs.
   - ✅ **Updated Experiment Configs**: Corrected provider naming in `experiments/multi_model_full_suite_poc1.yaml` (e.g., `meta-llama` instead of `meta`) to align with OpenRouter's model naming conventions.
   - **Follow-up work**: Run baseline experiments across diverse OpenRouter models (GPT-5, Claude 4.5, DeepSeek R1, Llama 4, Qwen 3, GLM 4.6, etc.) to establish quality benchmarks and monitor cross-provider performance characteristics.
