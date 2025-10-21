# Repository Guidelines
## Project Structure & Module Organization
- `src/generator/` holds config agents; keep orchestration in `core/` and reusable helpers under `tools/`.
- `src/evaluator/` owns scoring (`core/` for flows, `validators/` for policy, `reports/` for outputs); plan cross-cutting changes from here.
- Shared utilities live in `src/common/` and `src/utils/`; keep them dependency-light. Runtime artefacts go to `evaluation_reports/` and `tmp/` and should stay untracked.
- Entry scripts (`generator.py`, `evaluator.py`, `kind.sh`, `prepare-worker.sh`) should remain thin wrappers around module logic.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` sets up a Python 3.11+ virtualenv.
- `pip install -r requirements.txt` installs dependencies; rerun after editing `requirements.txt`.
- `python generator.py` runs the sample generation flow; adjust the `repo_url` constant or call `ConfigurationGenerator(...).generate(<url>)` from a REPL for ad-hoc targets.
- `python evaluator.py https://github.com/org/repo.git` scores a single repository; omit the URL to run the curated list and refresh reports in `evaluation_reports/`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents, type hints on public APIs, and snake_case modules; prefer Pydantic models for structured config.
- Give tools imperative names (`clone_repo`, `prepare_tree`) and keep agents small with clear logging via f-strings.
- Surface new knobs through `GeneratorConfig` so they pick up `.env` overrides, and document side effects in concise docstrings.

## Testing Guidelines
- Use `pytest`; mirror module paths under `tests/` (create it if missing) and name files `test_<module>.py`.
- Mock LLM, Docker, and Kubernetes clients to keep unit tests offline; reserve slow end-to-end checks for an `integration` marker.
- Capture expected artefact names or validator outputs so regressions are caught automatically, and note any manual `evaluation_reports/...` files in PRs.

## Commit & Pull Request Guidelines
- Git history shows short `TYPE: summary` subjects (e.g., `WIP: applying k8s`); follow the same prefixing and keep lines under 72 characters.
- Detail linked issues, test commands, and retained artefacts in the body before requesting review.
- PRs should include an overview, screenshots or report excerpts when outputs change, and call out required secrets or config updates.

## Security & Configuration Tips
- Load secrets via `.env`; `ConfigurationGenerator` reads `REGISTRY_URL`, `K8S_CLUSTER_IP`, and LLM keys through `dotenv`. Never commit the file.
- Clear `tmp/` after runs (`rm -rf tmp/*`) to avoid leaking cloned repos or manifests, and document any Kubernetes domain or namespace overrides.
