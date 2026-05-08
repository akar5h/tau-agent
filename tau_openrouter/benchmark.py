import json
import multiprocessing
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from tau_bench.envs import get_env
from tau_bench.run import display_metrics
from tau_bench.types import EnvRunResult, RESPOND_ACTION_NAME, RunConfig

from tau_openrouter.openai_agent import OpenAIChatReActAgent, OpenAIToolCallingAgent
from tau_openrouter.openai_compat import request_settings
from tau_openrouter.openai_user import install_user_patch
from tau_openrouter.kairos_setup import install_kairos, shutdown_kairos


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _last_agent_reply(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return None


class TracedEnv:
    def __init__(self, env: Any, tracer: Any) -> None:
        self._env = env
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        if action.name == RESPOND_ACTION_NAME or action.name not in self._env.tools_map:
            return self._env.step(action)
        with self._tracer.start_as_current_span(
            f"tool.{action.name}",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": action.name,
                "gen_ai.tool.call.arguments": _json_text(action.kwargs),
            },
        ) as span:
            result = self._env.step(action)
            span.set_attribute("gen_ai.tool.call.result", _json_text(result.observation))
            if isinstance(result.observation, str) and result.observation.startswith("Error:"):
                span.set_status(Status(StatusCode.ERROR, result.observation))
            return result


def run_benchmark(
    config: RunConfig,
    *,
    enable_kairos: bool = False,
    sleep_between_tasks_s: float = 0.0,
) -> list[EnvRunResult]:
    install_user_patch()
    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = (
        f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-"
        f"{config.temperature}_range_{config.start_index}-{config.end_index}_"
        f"user-{config.user_model.split('/')[-1]}-{config.user_strategy}_{time_str}.json"
    )
    os.makedirs(config.log_dir, exist_ok=True)

    if enable_kairos:
        install_kairos()
    tracer = trace.get_tracer(__name__) if enable_kairos else None

    print(f"Loading user with strategy: {config.user_strategy}")
    print(f"Agent model/provider: {config.model} via {config.model_provider}")
    print(f"User model/provider: {config.user_model} via {config.user_model_provider}")
    print(f"Agent request settings: {request_settings('TAU_BENCH_', config.temperature)}")
    print(f"User request settings: {request_settings('TAU_BENCH_USER_')}")
    env = get_env(
        config.env,
        user_strategy="human",
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    if config.agent_strategy == "tool-calling":
        agent = OpenAIToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        agent = OpenAIChatReActAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        agent = OpenAIChatReActAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    else:
        raise ValueError("few-shot is not implemented in the direct OpenAI runtime")
    end_index = len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    results: list[EnvRunResult] = []
    lock = multiprocessing.Lock()

    if config.task_ids:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")

    try:
        for trial in range(config.num_trials):
            idxs = config.task_ids or list(range(config.start_index, end_index))
            if config.shuffle:
                random.shuffle(idxs)

            def _run(idx: int) -> EnvRunResult:
                isolated_env = get_env(
                    config.env,
                    user_strategy=config.user_strategy,
                    user_model=config.user_model,
                    task_split=config.task_split,
                    user_provider=config.user_model_provider,
                    task_index=idx,
                )
                current_env = TracedEnv(isolated_env, tracer) if tracer is not None else isolated_env
                task = isolated_env.tasks[idx]
                print(f"Running task {idx}")
                try:
                    if tracer is None:
                        res = agent.solve(env=current_env, task_index=idx)
                    else:
                        with tracer.start_as_current_span(
                            "kairos.task",
                            attributes={
                                "kairos.agent.name": "tau_openrouter",
                                "kairos.business_op": f"tau_{config.env}",
                                "kairos.user_input": task.instruction,
                                "kairos.output_type": "text",
                                "kairos.metadata.task_id": idx,
                                "kairos.metadata.trial": trial,
                                "kairos.metadata.task_split": config.task_split,
                                "kairos.metadata.agent_strategy": config.agent_strategy,
                                "kairos.metadata.model": config.model,
                                "kairos.metadata.user_model": config.user_model,
                            },
                        ) as span:
                            try:
                                res = agent.solve(env=current_env, task_index=idx)
                            except Exception as exc:
                                span.record_exception(exc)
                                span.set_status(Status(StatusCode.ERROR, str(exc)))
                                raise
                            final_output = _last_agent_reply(res.messages)
                            if final_output is not None:
                                span.set_attribute("kairos.final_output", final_output)
                            span.set_attribute("kairos.metadata.reward", res.reward)
                            span.set_attribute("kairos.terminal_status", "completed")
                            if res.total_cost is not None:
                                span.set_attribute("kairos.metadata.agent_cost", res.total_cost)
                            user_cost = res.info.get("user_cost")
                            if user_cost is not None:
                                span.set_attribute("kairos.metadata.user_cost", user_cost)
                            span.set_status(Status(StatusCode.OK))
                    result = EnvRunResult(
                        task_id=idx,
                        reward=res.reward,
                        info=res.info,
                        traj=res.messages,
                        trial=trial,
                    )
                except Exception as exc:
                    result = EnvRunResult(
                        task_id=idx,
                        reward=0.0,
                        info={"error": str(exc), "traceback": traceback.format_exc()},
                        traj=[],
                        trial=trial,
                    )
                print("✅" if result.reward == 1 else "❌", f"task_id={idx}", result.info)
                print("-----")
                with lock:
                    data = []
                    if os.path.exists(ckpt_path):
                        with open(ckpt_path, "r", encoding="utf-8") as handle:
                            data = json.load(handle)
                    with open(ckpt_path, "w", encoding="utf-8") as handle:
                        json.dump(data + [result.model_dump()], handle, indent=2)
                if sleep_between_tasks_s > 0:
                    print(f"Sleeping {sleep_between_tasks_s}s before the next task")
                    time.sleep(sleep_between_tasks_s)
                return result

            with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
                results.extend(executor.map(_run, idxs))
    finally:
        if enable_kairos:
            shutdown_kairos()

    display_metrics(results)
    with open(ckpt_path, "w", encoding="utf-8") as handle:
        json.dump([result.model_dump() for result in results], handle, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results
