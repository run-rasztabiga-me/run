from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from ....generator.utils.llm import build_llm_kwargs
from ..pipeline import ValidationContext, ValidationState, ValidationStep, ValidationStepResult

def _extract_json_object(raw_text: str) -> Optional[Dict]:
    """Extract and parse JSON object from raw LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove fenced code block indicators
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


@dataclass
class LLMJudgeResult:
    score: Optional[float]
    summary: Optional[str]
    strengths: List[str]
    risks: List[str]
    artifacts_evaluated: List[str]

    def to_metadata(self) -> Dict[str, object]:
        return {
            "score": self.score,
            "summary": self.summary,
            "strengths": self.strengths,
            "risks": self.risks,
            "artifacts_evaluated": self.artifacts_evaluated,
        }


class BaseLLMJudgeStep(ValidationStep):
    """Base class providing shared plumbing for LLM-as-judge validation steps."""

    system_prompt: str
    artifact_type: str
    name: str

    def __init__(self) -> None:
        self._llm = None

    def run(self, state: ValidationState, context: ValidationContext) -> ValidationStepResult:
        if not context.config.enable_llm_judge:
            return ValidationStepResult(metadata={"skipped": True, "reason": "LLM judge disabled via configuration"})

        artifacts = self._gather_artifacts(state, context)
        if not artifacts:
            context.logger.debug("LLM judge step %s skipped: no %s artifacts found.", self.name, self.artifact_type)
            return ValidationStepResult(metadata={"skipped": True, "reason": f"No {self.artifact_type} artifacts available"})

        context.logger.info(
            "Running %s for repo %s on %d %s artifact(s).",
            self.name,
            context.run_context.repo_name,
            len(artifacts),
            self.artifact_type,
        )

        try:
            result = self._evaluate_artifacts(artifacts=artifacts, context=context)
            if result.score is not None:
                context.logger.info("%s score: %.2f (%s)", self.name, result.score, result.summary or "no summary")
            else:
                context.logger.warning("%s returned no numeric score; see raw response.", self.name)
            return ValidationStepResult(metadata=result.to_metadata())
        except ValueError as exc:
            context.logger.warning("Skipping %s: %s", self.name, exc)
            return ValidationStepResult(metadata={"skipped": True, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            context.logger.exception("Failed to execute %s", self.name)
            return ValidationStepResult(metadata={"skipped": True, "error": str(exc)})

    def _gather_artifacts(self, state: ValidationState, context: ValidationContext) -> List[Dict[str, str]]:
        """Collect artifacts for evaluation (path + content)."""
        raise NotImplementedError

    def _ensure_llm(self, context: ValidationContext):
        if self._llm is None:
            context.logger.debug(
                "Initialising LLM judge model (provider=%s, model=%s).",
                context.config.llm_judge_model_provider or context.config.model_provider,
                context.config.llm_judge_model_name or context.config.model_name,
            )
            llm_kwargs = build_llm_kwargs(
                context.config,
                logger=context.logger,
                provider_override=context.config.llm_judge_model_provider,
                model_override=context.config.llm_judge_model_name,
                temperature_override=context.config.llm_judge_temperature,
            )
            self._llm = init_chat_model(**llm_kwargs)
        return self._llm

    def _compose_user_prompt(self, artifacts: List[Dict[str, str]], context: ValidationContext) -> str:
        segments: List[str] = []
        for artifact in artifacts:
            segments.append(
                f"File: {artifact['path']}\n"
                "------------------------------\n"
                f"{artifact['content']}"
            )
        repo_name = context.run_context.repo_name
        return (
            f"Repository under review: {repo_name}\n"
            f"Evaluate the following {self.artifact_type} for production readiness.\n\n"
            + "\n\n".join(segments)
        )

    def _evaluate_artifacts(self, artifacts: List[Dict[str, str]], context: ValidationContext) -> LLMJudgeResult:
        llm = self._ensure_llm(context)
        user_prompt = self._compose_user_prompt(artifacts, context)
        context.logger.debug("%s prompt size: %d characters.", self.name, len(user_prompt))
        response = llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        if hasattr(response, "content"):
            raw_output = response.content
        else:
            raw_output = str(response)

        parsed = _extract_json_object(raw_output)
        score = None
        summary = None
        strengths: List[str] = []
        risks: List[str] = []

        if isinstance(parsed, dict):
            score = parsed.get("score")
            summary = parsed.get("summary") or parsed.get("verdict")
            strengths = parsed.get("strengths") or parsed.get("positives") or []
            risks = parsed.get("risks") or parsed.get("issues") or parsed.get("negatives") or []

            try:
                if score is not None:
                    score = float(score)
            except (TypeError, ValueError):
                score = None

            strengths = [str(item) for item in strengths] if isinstance(strengths, list) else [str(strengths)]
            risks = [str(item) for item in risks] if isinstance(risks, list) else [str(risks)]

        if parsed is None:
            context.logger.warning("%s: unable to parse JSON payload from LLM response.", self.name)

        result = LLMJudgeResult(
            score=score,
            summary=summary,
            strengths=strengths,
            risks=risks,
            artifacts_evaluated=[artifact["path"] for artifact in artifacts],
        )
        if parsed is None:
            result.summary = result.summary or raw_output[:200]

        return result


class DockerLLMJudgeStep(BaseLLMJudgeStep):
    """LLM-backed qualitative review for Dockerfiles."""

    name = "docker_llm_judge"
    artifact_type = "Dockerfile"
    system_prompt = (
        "You are an experienced DevOps engineer evaluating Dockerfiles generated by an automated agent. "
        "Score the Dockerfiles from 0 (unusable) to 100 (production ready) based on best practices, "
        "security hardening, maintainability, build efficiency, and alignment with Kubernetes deployment. "
        "Respond with compact JSON using the following schema: "
        '{"score": <number 0-100>, "summary": "<one sentence>", '
        '"strengths": ["<short bullet>", ...], "risks": ["<short bullet>", ...]}. '
        "Do not include markdown or extra commentary."
    )

    def _gather_artifacts(self, state: ValidationState, context: ValidationContext) -> List[Dict[str, str]]:
        artifacts: List[Dict[str, str]] = []
        for dockerfile_path in state.dockerfiles:
            read_result = context.workspace.read_file(dockerfile_path)
            if not read_result.success or read_result.content is None:
                context.logger.warning("Unable to read Dockerfile %s for LLM judge: %s", dockerfile_path, read_result.error)
                continue
            artifacts.append({"path": dockerfile_path, "content": read_result.content})
        return artifacts


class KubernetesLLMJudgeStep(BaseLLMJudgeStep):
    """LLM-backed qualitative review for Kubernetes manifests."""

    name = "k8s_llm_judge"
    artifact_type = "Kubernetes manifest"
    system_prompt = (
        "You are an experienced Kubernetes platform engineer reviewing manifests generated by an automated agent. "
        "Rate them from 0 (dangerous / non-functional) to 100 (production ready) considering correctness, "
        "security, resource hygiene, operational readiness, and maintainability. "
        "Respond with compact JSON using this schema: "
        '{"score": <number 0-100>, "summary": "<one sentence>", '
        '"strengths": ["<short bullet>", ...], "risks": ["<short bullet>", ...]}. '
        "Do not include markdown, explanations, or any text outside the JSON payload."
    )

    def _gather_artifacts(self, state: ValidationState, context: ValidationContext) -> List[Dict[str, str]]:
        artifacts: List[Dict[str, str]] = []
        for manifest_path in state.manifests:
            read_result = context.workspace.read_file(manifest_path)
            if not read_result.success or read_result.content is None:
                context.logger.warning("Unable to read Kubernetes manifest %s for LLM judge: %s", manifest_path, read_result.error)
                continue
            artifacts.append({"path": manifest_path, "content": read_result.content})
        return artifacts
