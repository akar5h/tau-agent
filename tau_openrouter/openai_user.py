from __future__ import annotations

import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tau_bench.envs.user import BaseUserSimulationEnv, UserStrategy

from tau_openrouter.openai_compat import (
    build_client,
    call_with_rate_limit_retry,
    chat_kwargs,
    env_int,
    log_api_error,
    request_settings,
    wait_for_rate_limit,
)


RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)


class OpenAIUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        self.client = build_client(provider)
        self.messages: list[dict[str, Any]] = []
        self.model = model
        self.total_cost = 0.0
        self.max_retries = env_int("TAU_BENCH_USER_RETRIES") or env_int("TAU_BENCH_RETRIES") or 0

    def generate_next_message(self, messages: list[dict[str, Any]]) -> str:
        kwargs = chat_kwargs("TAU_BENCH_USER_")
        settings = request_settings("TAU_BENCH_USER_")
        attempt = 0
        while True:
            try:
                wait_for_rate_limit("user simulator")
                res = call_with_rate_limit_retry(
                    "User simulator",
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **kwargs,
                    ),
                )
                message = res.choices[0].message
                self.messages.append(message.model_dump(exclude_none=True))
                return message.content or ""
            except RETRYABLE_ERRORS as exc:
                if not isinstance(exc, RateLimitError):
                    log_api_error("User simulator", exc)
                if attempt >= self.max_retries:
                    raise
                sleep_s = min(2 * (attempt + 1), 8)
                print(
                    f"User simulator request failed on attempt {attempt + 1}/{self.max_retries + 1} "
                    f"(model={self.model}, timeout={settings['timeout']}, max_tokens={settings['max_tokens']}, "
                    f"thinking={settings['thinking']}, prompt_messages={len(messages)}): {exc}. "
                    f"Retrying in {sleep_s}s"
                )
                time.sleep(sleep_s)
                attempt += 1

    def build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = (("\n\nInstruction: " + instruction + "\n") if instruction is not None else "")
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class OpenAIReactUserSimulationEnv(OpenAIUserSimulationEnv):
    def build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = (("\n\nInstruction: " + instruction + "\n") if instruction is not None else "")
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:
Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""

    def parse_response(self, response: str) -> str:
        if "###STOP###" in response:
            return "###STOP###"
        if "User Response:" in response:
            return response.split("User Response:")[-1].strip()
        if "Thought:" in response:
            return response.split("Thought:")[-1].strip()
        raise ValueError(f"Invalid response format: {response}")

    def generate_next_message(self, messages: list[dict[str, Any]]) -> str:
        return self.parse_response(super().generate_next_message(messages))


def map_role_label(role: str) -> str:
    if role == "user":
        return "Customer"
    if role == "assistant":
        return "Agent"
    return role.capitalize()


def verify(provider: str, model: str, response: str, messages: list[dict[str, Any]]) -> bool:
    transcript = "\n".join(f"{map_role_label(message['role'])}: {message['content']}" for message in messages)
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).

# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    client = build_client(provider)
    wait_for_rate_limit("user verifier")
    res = call_with_rate_limit_retry(
        "User verifier",
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **chat_kwargs("TAU_BENCH_USER_"),
        ),
    )
    return "true" in (res.choices[0].message.content or "").lower()


class OpenAIVerifyUserSimulationEnv(OpenAIUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 3) -> None:
        super().__init__(model=model, provider=provider)
        self.provider = provider
        self.max_attempts = max_attempts

    def generate_next_message(self, messages: list[dict[str, Any]]) -> str:
        attempts = 0
        current = ""
        while attempts < self.max_attempts:
            current = super().generate_next_message(messages)
            if verify(self.provider, self.model, current, messages):
                return current
            attempts += 1
        return current


def reflect(provider: str, model: str, response: str, messages: list[dict[str, Any]]) -> str:
    transcript = "\n".join(f"{map_role_label(message['role'])}: {message['content']}" for message in messages)
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.

# Transcript:
{transcript}

# Response:
{response}

# Format:

Reflection:
<the reflection>

Response:
<the response (this will be parsed and sent to the agent)>"""
    client = build_client(provider)
    wait_for_rate_limit("user reflector")
    res = call_with_rate_limit_retry(
        "User reflector",
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **chat_kwargs("TAU_BENCH_USER_"),
        ),
    )
    return (res.choices[0].message.content or "").split("Response:")[-1].strip()


class OpenAIReflectionUserSimulationEnv(OpenAIUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 2) -> None:
        super().__init__(model=model, provider=provider)
        self.provider = provider
        self.max_attempts = max_attempts

    def generate_next_message(self, messages: list[dict[str, Any]]) -> str:
        current_messages = messages.copy()
        initial = super().generate_next_message(current_messages)
        if verify(self.provider, self.model, initial, current_messages):
            return initial
        attempts = 1
        while attempts < self.max_attempts:
            new_message = reflect(self.provider, self.model, initial, current_messages)
            current_messages.append({"role": "user", "content": new_message})
            new_response = super().generate_next_message(current_messages)
            if verify(self.provider, self.model, new_response, current_messages):
                return new_response
            attempts += 1
        return initial


def load_user(user_strategy: str | UserStrategy, model: str | None = "gpt-4o", provider: str | None = None) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        from tau_bench.envs.user import HumanUserSimulationEnv

        return HumanUserSimulationEnv()
    if model is None or provider is None:
        raise ValueError("LLM-backed user strategies require both model and provider")
    if user_strategy == UserStrategy.LLM:
        return OpenAIUserSimulationEnv(model=model, provider=provider)
    if user_strategy == UserStrategy.REACT:
        return OpenAIReactUserSimulationEnv(model=model, provider=provider)
    if user_strategy == UserStrategy.VERIFY:
        return OpenAIVerifyUserSimulationEnv(model=model, provider=provider)
    if user_strategy == UserStrategy.REFLECTION:
        return OpenAIReflectionUserSimulationEnv(model=model, provider=provider)
    raise ValueError(f"Unknown user strategy {user_strategy}")


def install_user_patch() -> None:
    import tau_bench.envs.base as env_base
    import tau_bench.envs.user as env_user

    env_base.load_user = load_user
    env_user.load_user = load_user
