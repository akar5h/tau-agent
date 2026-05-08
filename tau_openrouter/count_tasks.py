import argparse

from dotenv import load_dotenv
from tau_bench.envs import get_env

from tau_openrouter.openai_user import install_user_patch
from tau_openrouter.run import configure_provider_env, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Count available tau-bench tasks for one env/split.")
    parser.add_argument("--provider", choices=["openrouter", "openai"], required=True)
    parser.add_argument("--env", choices=["retail", "airline"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--user-model")
    parser.add_argument("--user-strategy", default="llm")
    parser.add_argument("--task-split", choices=["train", "test", "dev"], default="test")
    args = parser.parse_args()

    load_dotenv()
    configure_provider_env(args.provider)
    install_user_patch()
    env = get_env(
        args.env,
        user_strategy="human",
        user_model=resolve_model(args.provider, args.user_model or args.model),
        user_provider=args.provider,
        task_split=args.task_split,
    )
    print(len(env.tasks))


if __name__ == "__main__":
    main()
