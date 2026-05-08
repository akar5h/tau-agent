import argparse
import os

from dotenv import load_dotenv
from tau_bench.types import RunConfig

from tau_openrouter.benchmark import run_benchmark


def env_or(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_provider_env(provider: str) -> None:
    if provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is required for provider=openrouter.")
        return
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required for provider=openai.")
        if os.getenv("OPENAI_API_BASE"):
            os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]
        return
    raise SystemExit(f"Unsupported provider: {provider}")


def resolve_model(provider: str, model: str) -> str:
    del provider
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tau-bench with OpenRouter or an OpenAI-compatible base URL.")
    parser.add_argument("--provider", choices=["openrouter", "openai"], default=env_or("TAU_BENCH_PROVIDER", "openrouter"))
    parser.add_argument("--env", choices=["retail", "airline"], default="retail")
    parser.add_argument("--model", default=env_or("TAU_BENCH_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--user-model", default=env_or("TAU_BENCH_USER_MODEL"))
    parser.add_argument(
        "--agent-strategy",
        choices=["tool-calling", "act", "react", "few-shot"],
        default=env_or("TAU_BENCH_AGENT_STRATEGY", "tool-calling"),
    )
    parser.add_argument(
        "--user-strategy",
        choices=["human", "llm", "react", "verify", "reflection"],
        default=env_or("TAU_BENCH_USER_STRATEGY", "llm"),
    )
    parser.add_argument("--task-split", choices=["train", "test", "dev"], default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=int(env_or("TAU_BENCH_MAX_CONCURRENCY", "1")))
    parser.add_argument(
        "--sleep-between-tasks",
        type=float,
        default=float(env_or("TAU_BENCH_SLEEP_BETWEEN_TASKS", "0") or "0"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--log-dir", default=env_or("TAU_BENCH_RESULTS_DIR", "results"))
    parser.add_argument("--few-shot-displays-path")
    parser.add_argument("--enable-kairos", action="store_true", default=env_flag("TAU_BENCH_ENABLE_KAIROS"))
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    configure_provider_env(args.provider)
    if args.first_n is not None:
        if args.task_ids:
            raise SystemExit("--first-n and --task-ids cannot be used together.")
        args.start_index = 0
        args.end_index = args.first_n
    user_model = args.user_model or args.model
    return RunConfig(
        model_provider=args.provider,
        user_model_provider=args.provider,
        model=resolve_model(args.provider, args.model),
        user_model=resolve_model(args.provider, user_model),
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_benchmark(
        build_config(args),
        enable_kairos=args.enable_kairos,
        sleep_between_tasks_s=args.sleep_between_tasks,
    )


if __name__ == "__main__":
    main()
