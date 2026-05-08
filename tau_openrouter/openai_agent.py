import json
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, RESPOND_ACTION_FIELD_NAME, RESPOND_ACTION_NAME, SolveResult

from tau_openrouter.openai_compat import (
    build_client,
    call_with_rate_limit_retry,
    chat_kwargs,
    log_api_error,
    wait_for_rate_limit,
)


RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)


def message_to_action(message: dict[str, Any]) -> Action:
    tool_calls = message.get("tool_calls") or []
    if tool_calls and tool_calls[0].get("function") is not None:
        tool_call = tool_calls[0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content") or ""})


class OpenAIToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.client = build_client(provider)
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.temperature = temperature

    def solve(self, env: Env, task_index: int | None = None, max_num_steps: int = 30) -> SolveResult:
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            try:
                wait_for_rate_limit("agent")
                res = call_with_rate_limit_retry(
                    "Agent",
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools_info,
                        **chat_kwargs("TAU_BENCH_", self.temperature),
                    ),
                )
            except RETRYABLE_ERRORS as exc:
                if not isinstance(exc, RateLimitError):
                    log_api_error("Agent", exc)
                raise
            next_message = res.choices[0].message.model_dump(exclude_none=True)
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend([next_message, {"role": "user", "content": env_response.observation}])
            if env_response.done:
                break
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=None)


REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}
"""


ACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}
"""


class OpenAIChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        self.client = build_client(provider)
        self.model = model
        self.temperature = temperature

    def generate_next_step(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], Action]:
        try:
            wait_for_rate_limit("agent")
            res = call_with_rate_limit_retry(
                "Agent",
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **chat_kwargs("TAU_BENCH_", self.temperature),
                ),
            )
        except RETRYABLE_ERRORS as exc:
            if not isinstance(exc, RateLimitError):
                log_api_error("Agent", exc)
            raise
        message = res.choices[0].message
        content = message.content or ""
        action_str = content.split("Action:")[-1].strip()
        try:
            parsed = json.loads(action_str)
        except json.JSONDecodeError:
            parsed = {"name": RESPOND_ACTION_NAME, "arguments": {RESPOND_ACTION_FIELD_NAME: action_str}}
        return message.model_dump(exclude_none=True), Action(name=parsed["name"], kwargs=parsed["arguments"])

    def solve(self, env: Env, task_index: int | None = None, max_num_steps: int = 30) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        info: dict[str, Any] = {}
        for _ in range(max_num_steps):
            message, action = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend([message, {"role": "user", "content": obs}])
            if response.done:
                break
        return SolveResult(messages=messages, reward=reward, info=info)
