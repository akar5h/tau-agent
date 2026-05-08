#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_tau.sh [options]

Options:
  --provider <openai|openrouter>   LiteLLM provider path. Default: value from .env
  --env <retail|airline>           Tau domain. Default: retail
  --model <model_id>               Exact model id to run
  --user-model <model_id>          Optional user-simulator model id
  --task-split <split>             Task split. Default: test
  --cases <n>                      Run the first n tasks as a smoke/subset run
  --all                            Run the full available task set
  --agent-strategy <strategy>      tool-calling, act, react, few-shot
  --user-strategy <strategy>       human, llm, react, verify, reflection
  --max-concurrency <n>            Parallel task count
  --sleep-between-tasks <sec>      Cooldown between finished tasks
  --enable-kairos                  Turn on Kairos + OpenLLMetry
  --log-dir <dir>                  Results directory override
  --help                           Show this help

Examples:
  ./scripts/run_tau.sh --cases 1
  ./scripts/run_tau.sh --provider openai --model meta/llama-3.1-70b-instruct --cases 2
  ./scripts/run_tau.sh --env airline --all --enable-kairos
EOF
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi
  if [ -x "$VENV_DIR/bin/uv" ]; then
    UV_BIN="$VENV_DIR/bin/uv"
    return
  fi
  mkdir -p "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$VENV_DIR/bin/pip" install uv
  if [ -x "$VENV_DIR/bin/uv" ]; then
    UV_BIN="$VENV_DIR/bin/uv"
    return
  fi
  echo "Unable to find uv after installation. Install uv globally and rerun."
  exit 1
}

ensure_env_file() {
  if [ -f "$ROOT_DIR/.env" ]; then
    return
  fi
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "Created $ROOT_DIR/.env from .env.example. Fill in your API key and model id, then rerun."
  exit 1
}

load_env_file() {
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
}

validate_effective_config() {
  if [ -z "${EFFECTIVE_MODEL:-}" ] || [ "$EFFECTIVE_MODEL" = "paste_exact_model_id_here" ]; then
    echo "Set a real model id in .env or pass --model."
    exit 1
  fi
  if [ "$EFFECTIVE_PROVIDER" = "openai" ]; then
    if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "replace_with_your_openai_compatible_api_key" ]; then
      echo "Set OPENAI_API_KEY in .env for provider=openai."
      exit 1
    fi
    return
  fi
  if [ -z "${OPENROUTER_API_KEY:-}" ] || [ "$OPENROUTER_API_KEY" = "replace_with_your_openrouter_api_key" ]; then
    echo "Set OPENROUTER_API_KEY in .env for provider=openrouter."
    exit 1
  fi
}

PROVIDER=""
ENV_NAME="retail"
MODEL=""
USER_MODEL=""
TASK_SPLIT="test"
CASES=""
RUN_ALL=0
AGENT_STRATEGY=""
USER_STRATEGY=""
MAX_CONCURRENCY=""
SLEEP_BETWEEN_TASKS=""
ENABLE_KAIROS=0
LOG_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --provider) PROVIDER="${2:?}"; shift 2 ;;
    --env) ENV_NAME="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --user-model) USER_MODEL="${2:?}"; shift 2 ;;
    --task-split) TASK_SPLIT="${2:?}"; shift 2 ;;
    --cases) CASES="${2:?}"; shift 2 ;;
    --all) RUN_ALL=1; shift ;;
    --agent-strategy) AGENT_STRATEGY="${2:?}"; shift 2 ;;
    --user-strategy) USER_STRATEGY="${2:?}"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="${2:?}"; shift 2 ;;
    --sleep-between-tasks) SLEEP_BETWEEN_TASKS="${2:?}"; shift 2 ;;
    --enable-kairos) ENABLE_KAIROS=1; shift ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [ -n "$CASES" ] && [ "$RUN_ALL" -eq 1 ]; then
  echo "Use either --cases N or --all, not both."
  exit 1
fi

if [ -z "$CASES" ] && [ "$RUN_ALL" -eq 0 ]; then
  CASES="1"
fi

ensure_env_file
load_env_file

EFFECTIVE_PROVIDER="${PROVIDER:-${TAU_BENCH_PROVIDER:-openai}}"
EFFECTIVE_MODEL="${MODEL:-${TAU_BENCH_MODEL:-}}"
EFFECTIVE_USER_MODEL="${USER_MODEL:-${TAU_BENCH_USER_MODEL:-}}"
EFFECTIVE_USER_STRATEGY="${USER_STRATEGY:-${TAU_BENCH_USER_STRATEGY:-llm}}"

validate_effective_config
ensure_uv

cd "$ROOT_DIR"
"$UV_BIN" sync

TASK_COUNT="$("$UV_BIN" run python -m tau_openrouter.count_tasks \
  --provider "$EFFECTIVE_PROVIDER" \
  --env "$ENV_NAME" \
  --model "$EFFECTIVE_MODEL" \
  --user-model "${EFFECTIVE_USER_MODEL:-$EFFECTIVE_MODEL}" \
  --user-strategy "$EFFECTIVE_USER_STRATEGY" \
  --task-split "$TASK_SPLIT")"

if [ -n "$CASES" ]; then
  case "$CASES" in
    ''|*[!0-9]*) echo "--cases must be a positive integer."; exit 1 ;;
  esac
  if [ "$CASES" -lt 1 ]; then
    echo "--cases must be at least 1."
    exit 1
  fi
  if [ "$CASES" -gt "$TASK_COUNT" ]; then
    echo "--cases $CASES exceeds available tasks for env=$ENV_NAME split=$TASK_SPLIT (count=$TASK_COUNT)."
    exit 1
  fi
fi

CMD=( "$UV_BIN" run tau-openrouter --env "$ENV_NAME" --task-split "$TASK_SPLIT" )

CMD+=( --provider "$EFFECTIVE_PROVIDER" --model "$EFFECTIVE_MODEL" )
if [ -n "$EFFECTIVE_USER_MODEL" ]; then
  CMD+=( --user-model "$EFFECTIVE_USER_MODEL" )
fi
if [ -n "$AGENT_STRATEGY" ]; then
  CMD+=( --agent-strategy "$AGENT_STRATEGY" )
fi
CMD+=( --user-strategy "$EFFECTIVE_USER_STRATEGY" )
if [ -n "$MAX_CONCURRENCY" ]; then
  CMD+=( --max-concurrency "$MAX_CONCURRENCY" )
fi
if [ -n "$SLEEP_BETWEEN_TASKS" ]; then
  CMD+=( --sleep-between-tasks "$SLEEP_BETWEEN_TASKS" )
fi
if [ -n "$LOG_DIR" ]; then
  CMD+=( --log-dir "$LOG_DIR" )
fi
if [ "$ENABLE_KAIROS" -eq 1 ]; then
  CMD+=( --enable-kairos )
fi
if [ "$RUN_ALL" -eq 0 ]; then
  CMD+=( --first-n "$CASES" )
fi

echo "Task count for env=$ENV_NAME split=$TASK_SPLIT: $TASK_COUNT"
if [ "$RUN_ALL" -eq 0 ]; then
  echo "Running smoke/subset with first $CASES task(s)."
else
  echo "Running full task set."
fi
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
