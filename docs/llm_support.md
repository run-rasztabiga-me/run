# LLM Provider Support Checklist (2025 Roadmap)

The following models are slated for evaluation runs. This document tracks the integration steps required for each provider so the generator/evaluator stack can invoke them via `langchain.chat_models.init_chat_model`.

| Model | Provider string | Required LangChain package | Environment variables / credentials | Notes |
|-------|-----------------|----------------------------|-------------------------------------|-------|
| GPT-5 | `openai` | `langchain-openai` (already pinned) | `OPENAI_API_KEY` | Upgrade `openai`/`langchain-openai` to a release that exposes the `gpt-5` family once GA. |
| GPT-5 Mini | `openai` | `langchain-openai` | `OPENAI_API_KEY` | Same as above; confirm token pricing + max context for config defaults. |
| Claude Sonnet 4.5 | `anthropic` | `langchain-anthropic` (already pinned) | `ANTHROPIC_API_KEY` | Ensure package version ≥ release supporting `claude-sonnet-4.5`. |
| Claude Haiku 4.5 | `anthropic` | `langchain-anthropic` | `ANTHROPIC_API_KEY` | Already referenced in experiments; verify new revision ID. |
| Claude Opus 4.1 | `anthropic` | `langchain-anthropic` | `ANTHROPIC_API_KEY` | Update allowed max tokens / cost guardrails before enabling in prod. |
| Qwen 3 | `qwen` (TBC) | `langchain-qwen` / `langchain-dashscope` (confirm) | `QWEN_API_KEY` (DashScope) | Validate provider string + required base URL; add optional config overrides (region). |
| DeepSeek R1 | `deepseek` | `langchain-deepseek` | `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL` | Package introduced in LangChain ≥0.3.19; bump requirements accordingly. |
| DeepSeek V3.2 | `deepseek` | `langchain-deepseek` | `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL` | Same as R1; confirm that model identifier matches API docs. |
| GLM 4.6 | `zhipuai` | `langchain-zhipuai` | `ZHIPUAI_API_KEY`, optional `ZHIPUAI_API_SECRET` | Add dependency + env template entries; watch for region-specific endpoints. |
| Mistral Medium | `mistralai` | `langchain-mistralai` | `MISTRAL_API_KEY` | Evaluate whether tool calling requires extra kwargs (e.g., `safe_prompt`). |
| Llama 4 Maverick | `meta` (TBC) / `huggingface`/`together` | TBD (likely `langchain-meta` or `langchain-huggingface`) | `META_LLM_API_KEY` or provider-specific token | Determine official API surface (Meta AI, Groq, Together) before wiring provider string. |
| Llama 4 Scout | `meta` (TBC) / `huggingface`/`together` | TBD | `META_LLM_API_KEY` | Same as Maverick—confirm how to address via LangChain once endpoints ship. |

## Action Items

1. **Dependencies**
   - Add the missing provider packages (`langchain-deepseek`, `langchain-mistralai`, `langchain-zhipuai`, `langchain-qwen`/`langchain-dashscope`, plus any Meta/Llama integration once published) to `requirements.txt`.
   - Track minimum versions for existing providers (OpenAI, Anthropic) supporting the new model IDs.

2. **Environment & Configuration**
   - `.env.template` now contains placeholders for new API keys (`DEEPSEEK_API_KEY`, `QWEN_API_KEY`, `ZHIPUAI_API_KEY`, `MISTRAL_API_KEY`, `META_LLM_API_KEY`). Update onboarding docs/README once concrete provider instructions are confirmed.
   - Extend `GeneratorConfig` overrides if certain providers need extra kwargs (e.g., `max_output_tokens`, custom base URLs).

3. **Runtime Validations**
   - Some providers still lack deterministic seeding. Ensure `ConfigurationAgent`’s seed handling stays provider-aware (currently only applied to OpenAI).
   - Add smoke tests (or mocked unit tests) per provider to ensure `init_chat_model` resolves successfully with the new identifiers.

4. **Experiment Coverage**
   - `experiments/multi_model_full_suite_poc1.yaml` enumerates all target models for poc1. Once integrations are validated, schedule the experiment and capture baseline metrics for regression tracking.

5. **Documentation**
   - Update README/AGENTS.md when the integrations are live so contributors know which environment variables and dependencies are mandatory.

Use this checklist to drive the implementation PR(s); as provider SDKs ship, replace any “TBC” placeholders with concrete package names, env vars, and configuration snippets.
