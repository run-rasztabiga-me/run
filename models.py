import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from prompts import IMPORTANT_FILES_PROMPT, DOCKERFILE_PROMPT, K8S_CONFIG_PROMPT

load_dotenv()


def get_model(model_name: str) -> 'Model':
    # TODO
    return OpenRouterModel(model_name)
    # if model_name == "gpt-4o-mini":
    #     return OpenAIModel(model_name)
    # if model_name == "gpt-4o":
    #     return OpenAIModel(model_name)
    # else:
    #     return None


class Model(ABC):
    @abstractmethod
    def ask_model(self, use_case: str, user_prompt: str) -> str:
        """
        Ask the model a question and get a response.
        
        Args:
            use_case: The type of task to perform (e.g., "get_important_files", "get_dockerfile", "get_k8s_config")
            user_prompt: The input data for the model
            
        Returns:
            str: The model's response
            
        Raises:
            ValueError: If use_case is not supported
            OpenAIError: If there's an error communicating with the API
        """
        pass


class OpenAIModel(Model):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def ask_model(self, use_case: str, user_prompt: str) -> str:
        try:
            system_prompt = self._get_system_prompt(use_case)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return completion.choices[0].message.content
        except OpenAIError as e:
            raise OpenAIError(f"Error communicating with OpenAI API: {str(e)}")

    def _get_system_prompt(self, use_case: str) -> str:
        prompts = {
            "get_important_files": IMPORTANT_FILES_PROMPT,
            "get_dockerfile": DOCKERFILE_PROMPT,
            "get_k8s_config": K8S_CONFIG_PROMPT
        }
        if use_case not in prompts:
            raise ValueError(f"Unknown use case: {use_case}")
        return prompts[use_case]


class OpenRouterModel(Model):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def ask_model(self, use_case: str, user_prompt: str) -> str:
        try:
            system_prompt = self._get_system_prompt(use_case)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        except OpenAIError as e:
            raise OpenAIError(f"Error communicating with OpenRouter API: {str(e)}")

    def _get_system_prompt(self, use_case: str) -> str:
        prompts = {
            "get_important_files": IMPORTANT_FILES_PROMPT,
            "get_dockerfile": DOCKERFILE_PROMPT,
            "get_k8s_config": K8S_CONFIG_PROMPT
        }
        if use_case not in prompts:
            raise ValueError(f"Unknown use case: {use_case}")
        return prompts[use_case]

    def __str__(self) -> str:
        return "OpenRouterModel model: " + self.model
