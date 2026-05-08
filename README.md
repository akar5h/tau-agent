# tau-agent

Minimal `tau-bench` runner with:

- upstream `tau-bench` tasks, envs, tools, and reward computation
- direct OpenAI-compatible client calls for both the agent and the LLM user simulator
- support for NVIDIA Integrate and OpenRouter
- optional Phoenix/OpenTelemetry tracing for Kairos prototyping

This repo is meant to stay close to `tau-bench` while giving you a practical local harness for running retail and airline tasks, collecting native result JSON, and optionally emitting traces.

## What This Repo Does

- Uses `tau-bench` for:
  - task loading
  - environment simulation
  - tool execution
  - reward calculation
- Uses the OpenAI Python SDK directly against:
  - NVIDIA Integrate via `OPENAI_API_BASE`
  - OpenRouter via `OPENROUTER_API_BASE`
- Supports these agent strategies:
  - `tool-calling`
  - `act`
  - `react`
- Supports these user simulator strategies:
  - `human`
  - `llm`
  - `react`
  - `verify`
  - `reflection`
- Writes:
  - native `tau-bench` result JSON under `results/`
  - optional Kairos/Phoenix trace artifacts under `data/live/`

## Task Splits

Current task counts in the installed `tau-bench` package:

- `retail train`: `500`
- `retail test`: `115`
- `airline test`: `50`

Good starting runs:

- easiest smoke run: `retail train`, first `10`
- first benchmark-ish run: `retail test`, first `10`
- next domain: `airline test`, first `10`

## Install

Requirements:

- Python `3.11` to `3.13`
- `uv`

Setup:

```bash
cp .env.example .env
uv sync
```

Bootstrap runner:

```bash
./scripts/run_tau.sh --cases 1
```

That script will:

- make sure `uv` is available
- create `.env` from `.env.example` if missing
- install dependencies with `uv sync`
- validate your provider/model config
- count available tasks for the chosen split
- run a subset or the full split

## Environment Variables

### Required provider credentials

Use one of these provider setups.

NVIDIA Integrate / any OpenAI-compatible endpoint:

```env
OPENAI_API_KEY=your_provider_key
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
TAU_BENCH_PROVIDER=openai
TAU_BENCH_MODEL=moonshotai/kimi-k2-instruct
TAU_BENCH_USER_MODEL=moonshotai/kimi-k2-instruct
```

OpenRouter:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
TAU_BENCH_PROVIDER=openrouter
TAU_BENCH_MODEL=moonshotai/kimi-k2
TAU_BENCH_USER_MODEL=moonshotai/kimi-k2
```

### Core run settings

- `TAU_BENCH_PROVIDER`
  - `openai` or `openrouter`
- `TAU_BENCH_MODEL`
  - agent model id
- `TAU_BENCH_USER_MODEL`
  - user simulator model id
  - if empty, the runner falls back to the agent model
- `TAU_BENCH_AGENT_STRATEGY`
  - `tool-calling`, `act`, or `react`
- `TAU_BENCH_USER_STRATEGY`
  - `human`, `llm`, `react`, `verify`, or `reflection`
- `TAU_BENCH_RESULTS_DIR`
  - default `results`

### Model request tuning

Agent-side:

- `TAU_BENCH_TOP_P`
- `TAU_BENCH_MAX_TOKENS`
- `TAU_BENCH_TIMEOUT`
- `TAU_BENCH_THINKING`
- `TAU_BENCH_REASONING_EFFORT`
- `TAU_BENCH_RETRIES`

User-simulator-side:

- `TAU_BENCH_USER_TOP_P`
- `TAU_BENCH_USER_MAX_TOKENS`
- `TAU_BENCH_USER_TIMEOUT`
- `TAU_BENCH_USER_THINKING`
- `TAU_BENCH_USER_REASONING_EFFORT`
- `TAU_BENCH_USER_RETRIES`

### Rate limiting and backoff

These are especially important for NVIDIA hosted endpoints.

- `TAU_BENCH_MAX_CONCURRENCY`
  - keep this at `1` unless you really know the provider can handle more
- `TAU_BENCH_SLEEP_BETWEEN_TASKS`
  - optional extra pause between completed tasks
- `TAU_BENCH_REQUESTS_PER_MINUTE`
  - shared limiter across all model calls
  - current recommended default here is `12`
- `TAU_BENCH_RATE_LIMIT_RETRIES`
  - `429` retry count
- `TAU_BENCH_RATE_LIMIT_BACKOFF_BASE`
  - exponential backoff base in seconds
- `TAU_BENCH_RATE_LIMIT_BACKOFF_MAX`
  - max backoff delay in seconds
- `TAU_BENCH_RATE_LIMIT_BACKOFF_JITTER`
  - jitter added to the backoff
- `TAU_BENCH_SDK_MAX_RETRIES`
  - OpenAI SDK retry count for transport-level retries

### Tracing / Phoenix / Kairos

- `TAU_BENCH_ENABLE_KAIROS`
  - `1` to turn tracing on
- `PHOENIX_OTLP_ENDPOINT`
  - full OTLP traces endpoint
  - default in this repo: `http://localhost:6006/v1/traces`
- `TAU_BENCH_KAIROS_RAW_DIR`
  - default `data/live/raw`
- `TAU_BENCH_KAIROS_NORMALIZED_DIR`
  - default `data/live/normalized`
- `TAU_BENCH_LOG_API_HEADERS`
  - if enabled, prints interesting response headers on API failures such as `429`

## Running the Benchmark

### One-task smoke run

```bash
./scripts/run_tau.sh --env retail --task-split train --cases 1 --max-concurrency 1
```

### First 10 easy retail tasks

```bash
./scripts/run_tau.sh --env retail --task-split train --cases 10 --max-concurrency 1
```

### First 10 harder retail test tasks

```bash
./scripts/run_tau.sh --env retail --task-split test --cases 10 --max-concurrency 1
```

### First 20 airline tasks

```bash
./scripts/run_tau.sh --env airline --task-split test --cases 20 --max-concurrency 1
```

### Explicit task ids

```bash
uv run tau-openrouter \
  --env airline \
  --task-split test \
  --provider openai \
  --model moonshotai/kimi-k2-instruct \
  --user-model moonshotai/kimi-k2-instruct \
  --user-strategy llm \
  --max-concurrency 1 \
  --task-ids 0 1 2 3 4 5 6 7 8 9
```

### Full split

```bash
./scripts/run_tau.sh --env airline --task-split test --all --max-concurrency 1
```

### Module form

```bash
uv run python -m tau_openrouter.run --env retail --task-split train --first-n 10
```

## Phoenix / OpenTelemetry / Kairos Setup

This repo does not run a Phoenix UI server by itself. It only emits OpenTelemetry traces to a Phoenix-compatible OTLP endpoint.

What the code does when `--enable-kairos` is enabled:

- checks whether Phoenix is reachable
- registers an OTLP exporter via `phoenix.otel.register(...)`
- explicitly instruments the OpenAI SDK with OpenInference
- creates manual `kairos.task` spans per benchmark task
- creates manual `tool.<name>` spans for tool execution

Current tracing implementation lives in:

- [tau_openrouter/kairos_setup.py](/Users/akarshgajbhiye/tau-agent/tau_openrouter/kairos_setup.py)
- [tau_openrouter/benchmark.py](/Users/akarshgajbhiye/tau-agent/tau_openrouter/benchmark.py)

### Start Phoenix locally

One simple local path from the official Phoenix docs is:

```bash
pip install arize-phoenix
phoenix serve
```

Phoenix serves the local UI at `http://localhost:6006` by default. Source:

- [Phoenix terminal deployment docs](https://arize.com/docs/phoenix/self-hosting/deployment-options/terminal)

This repo expects the OTLP traces endpoint to be:

```env
PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces
```

Phoenix/OpenInference background:

- Phoenix accepts traces over OpenTelemetry / OTLP
- Phoenix’s Python SDK docs describe the collector/base-url environment variable model
- this repo uses `phoenix.otel.register(...)` from `arize-phoenix-otel` and passes the full OTLP traces URL directly

References:

- [Phoenix overview](https://arize.com/docs/phoenix)
- [Phoenix Python SDK docs](https://arize.com/docs/phoenix/sdk-api-reference)

If Phoenix is up, the runner prints something like:

```text
Phoenix collector reachable at http://localhost:6006 (status=200)
OpenAI OpenInference instrumented: True
```

### Run with tracing enabled

```bash
./scripts/run_tau.sh --env retail --task-split train --cases 5 --enable-kairos
```

### Trace outputs

When tracing is enabled:

- native benchmark results still go to `results/`
- raw Kairos live events go to `data/live/raw/`
- normalized envelopes go to `data/live/normalized/`

## Results Format

Benchmark outputs are written in native `tau-bench` JSON format:

- `task_id`
- `reward`
- `info`
- `traj`
- `trial`

Example output path:

```text
results/tool-calling-kimi-k2-instruct-0.0_range_0-10_user-kimi-k2-instruct-llm_0508200941.json
```

## Notes on Rate Limits

Hosted NVIDIA endpoints are the main pain point for multi-step agent benchmarks.

What matters here:

- one task can generate many model requests
- both the agent and the LLM user simulator consume the same provider budget
- task-level sleeping is not enough by itself

This repo therefore includes:

- a shared request-per-minute limiter
- exponential backoff on `429`
- optional response-header logging on API failures

If you still see repeated `429` errors:

- reduce `TAU_BENCH_REQUESTS_PER_MINUTE`
- keep `TAU_BENCH_MAX_CONCURRENCY=1`
- consider using a cheaper or different user-simulator provider/model

## Current Limitations

- `few-shot` agent strategy is not implemented in this direct OpenAI runtime
- the runner currently uses one provider for both the agent and user simulator
- if you want hybrid provider routing, that needs a small code change

## Safe Publishing

This repo ignores:

- `.env`
- `.venv/`
- `results/`
- `data/`

So benchmark outputs and local secrets do not need to be pushed with the code.
