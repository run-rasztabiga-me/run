import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI

from prompts import IMPORTANT_FILES_PROMPT, DOCKERFILE_PROMPT, K8S_CONFIG_PROMPT

load_dotenv()


def get_model(model_name):
    # TODO
    return OpenRouterModel(model_name)
    # if model_name == "gpt-4o-mini":
    #     return OpenAIModel(model_name)
    # if model_name == "gpt-4o":
    #     return OpenAIModel(model_name)
    # else:
    #     return None


class Model(ABC):
    def ask_model(self, use_case, user_prompt):
        prompt = f"{user_prompt}"
        return self.ask_model_internal(prompt, use_case)

    @abstractmethod
    def ask_model_internal(self, prompt, use_case):
        pass


class OpenAIModel(Model):
    def __init__(self, model):
        self.model = model
        self.client = OpenAI()

    def ask_model_internal(self, prompt, use_case):
        system_prompt = self._get_system_prompt(use_case)
        completion = self.client.chat.completions.create(model=self.model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        return completion.choices[0].message.content

    def _get_system_prompt(self, use_case):
        if use_case == "get_important_files":
            return IMPORTANT_FILES_PROMPT
        elif use_case == "get_dockerfile":
            return DOCKERFILE_PROMPT
        elif use_case == "get_k8s_config":
            return K8S_CONFIG_PROMPT
        else:
            raise ValueError(f"Unknown use case: {use_case}")


class OpenRouterModel(Model):
    def __init__(self, model):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def ask_model_internal(self, prompt, use_case):
        system_prompt = self._get_system_prompt(use_case)
        completion = self.client.chat.completions.create(model=self.model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ], response_format={"type": "json_object"})
        return completion.choices[0].message.content

    def _get_system_prompt(self, use_case):
        if use_case == "get_important_files":
            return IMPORTANT_FILES_PROMPT
        elif use_case == "get_dockerfile":
            return DOCKERFILE_PROMPT
        elif use_case == "get_k8s_config":
            return K8S_CONFIG_PROMPT
        else:
            raise ValueError(f"Unknown use case: {use_case}")

    def __str__(self):
        return "OpenRouterModel model: " + self.model
