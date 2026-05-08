# tau-bench OpenRouter baseline

Minimal scaffolding for running the upstream `tau-bench` benchmark through LiteLLM. It supports OpenRouter and OpenAI-compatible endpoints with a custom base URL, including NVIDIA Integrate. By default this runs without any extra tracing. If you pass `--enable-kairos`, the wrapper adds Kairos + OpenLLMetry around each task while keeping tau-bench's native task loading, user simulation, reward computation, and result format.

## Install

1. Copy `.env.example` to `.env`.
2. Set `OPENROUTER_API_KEY` in `.env`.
3. Install dependencies:

```bash
uv sync
```

## Run

One command bootstrap runner:

```bash
./scripts/run_tau.sh --cases 1
```

That script will:

- create `.venv` if needed
- install `uv` if missing
- run `uv sync`
- validate `.env`
- check the available task count for the selected domain/split
- run either a smoke subset with `--cases N` or the full set with `--all`

Run the first 10 retail tasks with `gpt-4o-mini`:

```bash
uv run tau-openrouter --env retail --model openai/gpt-4o-mini --first-n 10
```

Run against an OpenAI-compatible endpoint like NVIDIA Integrate:

```bash
uv run tau-openrouter --provider openai --env retail --model <paste-exact-model-id> --first-n 10
```

Using the bootstrap script with the same provider/model:

```bash
./scripts/run_tau.sh --provider openai --model <paste-exact-model-id> --cases 2
```

Run a single retail task:

```bash
uv run tau-openrouter --env retail --model openai/gpt-4o-mini --task-ids 0
```

Run the full airline domain:

```bash
uv run tau-openrouter --env airline --model openai/gpt-4o-mini
```

Enable Kairos normalization output for a short run:

```bash
uv run tau-openrouter --env retail --model openai/gpt-4o-mini --first-n 2 --enable-kairos
```

Use a different OpenRouter-backed model:

```bash
uv run tau-openrouter --env retail --model anthropic/claude-3.5-sonnet --first-n 10
uv run tau-openrouter --env retail --model deepseek/deepseek-chat-v3 --first-n 10
```

Equivalent module form:

```bash
uv run python -m tau_openrouter.run --env retail --model openai/gpt-4o-mini --first-n 10
```

## Model selection

- `--provider` chooses the LiteLLM provider path. Use `openrouter` for OpenRouter and `openai` for OpenAI-compatible endpoints.
- `--model` sets the agent model.
- `--user-model` sets the user simulator model. If omitted, it defaults to `--model`.
- You can also set defaults through `.env` with `TAU_BENCH_PROVIDER`, `TAU_BENCH_MODEL`, and `TAU_BENCH_USER_MODEL`.
- Pass model slugs in OpenRouter form without the transport prefix, for example:
  - `openai/gpt-4o-mini`
  - `anthropic/claude-3.5-sonnet`
  - `deepseek/deepseek-chat-v3`

The wrapper converts these to the LiteLLM form expected by upstream `tau-bench`, for example `openrouter/openai/gpt-4o-mini`.

For OpenAI-compatible endpoints, paste the exact model ID your endpoint expects into `TAU_BENCH_MODEL` or `--model`. The wrapper does not rewrite it.

For NVIDIA Integrate specifically, set:

```bash
OPENAI_API_KEY=...
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
TAU_BENCH_PROVIDER=openai
TAU_BENCH_MODEL=<paste-exact-model-id>
```

Then run:

```bash
./scripts/run_tau.sh --cases 1
```

## Results

Results land in `results/` by default, or the directory passed with `--log-dir`. The JSON file is written by upstream `tau-bench` and keeps its native schema: a JSON array of `EnvRunResult` records with `task_id`, `reward`, `info`, `traj`, and `trial`.

When Kairos is enabled:

- raw events land in `data/live/raw/` by default
- normalized envelopes land in `data/live/normalized/` by default
- both paths are configurable with `--kairos-raw-dir` and `--kairos-normalized-dir`

Bootstrap script examples:

```bash
./scripts/run_tau.sh --cases 1
./scripts/run_tau.sh --cases 5 --enable-kairos
./scripts/run_tau.sh --env airline --all
```

## Notes

- This targets the original `tau-bench` repository you requested. Upstream warns that newer fixed tasks live in `tau2-bench`, but this scaffold intentionally stays on `tau-bench`.
- Kairos is opt-in. Without `--enable-kairos`, no extra tracing path is installed.
