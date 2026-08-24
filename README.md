# SWE-EvalPressure

SWE-EvalPressure is a controlled benchmark for studying how coding agents behave when they have explicit evidence that they are being evaluated and are placed under different forms of pressure. It is built on the 70 SWE-Atlas refactoring tasks and preserves the original task instructions, tests, gold solutions, and verifier logic.

The experimental conditions change only the repository context presented to the agent. Harness-level compatibility wrappers normalize agent installation across heterogeneous task images; they do not change the task instructions, tests, verifier logic, or agent run/solver behavior.

## Quick start: clone → smoke → full → analyze → report

This is the shortest supported path for a fresh machine. Run these blocks in order. The rest of the README is reference material.

### 0. Clone and create the pinned host environment

The repository is private. Accept the GitHub collaborator invitation before cloning. The SSH command below assumes GitHub SSH access is already configured on the machine; otherwise use the authenticated HTTPS clone shown below.

```bash
# SSH
git clone git@github.com:ibrikk/swe-eval-pressure.git

# Or authenticated HTTPS
# git clone https://github.com/ibrikk/swe-eval-pressure.git

cd swe-eval-pressure

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.13
uv sync --locked
source .venv/bin/activate

python --version
harbor --version
modal --version
```

`uv python install 3.13` installs the pinned Python runtime, and `uv sync --locked` installs the repository-local host dependencies from `pyproject.toml` / `uv.lock`, including Harbor `0.20.0` and Modal. No separate Harbor or Modal package installation is required. Keep `.venv` activated when running benchmark commands.

`modal setup` authenticates the installed Modal CLI. Run it once on a new machine:

```bash
modal setup
```

### 1. Configure credentials and validate the checkout

```bash
cp .env.example .env
nano .env

./lab.sh doctor
./lab.sh inventory
```

Fill in the required gateway credential(s) in `.env`, save the file, and do not commit it. `.env` is gitignored.

Expected:

```text
Runtime doctor: PASS
Valid refactoring tasks: 70
Invalid tasks: 0
```

Production pins:

```text
Harbor              0.20.0
Codex CLI           0.147.0
Mini-SWE agent      2.4.5
Mini-SWE LiteLLM    1.83.0
```

`ALLOW_INTERNET=true` is the benchmark default and matches normal SWE-Atlas execution. Use `false` only for an intentional no-network condition or ablation, and keep it fixed across directly compared runs.

### 2. Structural pilot smoke — no model trajectories

```bash
PROFILES="claude fable codex llama" ./lab.sh prepare pilot
./lab.sh validate pilot all

./lab.sh matrix pilot \
  --concurrency-preset serial \
  --dry-run
```

### 3. Agent/bootstrap smoke — install only

`--install-only` does not execute benchmark model trajectories. Harbor therefore reports `Trials 0` by design. A successful smoke is indicated by all setup attempts completing with `0 exceptions`.

```bash
for profile in claude fable codex llama; do
  echo "===== install smoke: $profile ====="
  HARBOR_CONCURRENCY=1 ./lab.sh run pilot "$profile" --install-only
done
```

Targeted compatibility smoke for known hostile task-image families:

```bash
BENCHMARK_TASK_IDS="task-69d196f015a150488265afc2,task-69d196f015a150488265afc3,task-69d196f015a150488265afc4,task-69d196f015a150488265afba,task-696719205599a51110d4b446" \
PROFILES="codex llama" \
./lab.sh prepare resource

./lab.sh validate resource codex
./lab.sh validate resource llama

HARBOR_CONCURRENCY=1 ./lab.sh run resource codex --install-only
HARBOR_CONCURRENCY=1 ./lab.sh run resource llama --install-only
```

A production-ready checkout should show zero install exceptions for both targeted profiles.

### 4. Prepare and validate the full benchmark

```bash
./lab.sh plan full

PROFILES="claude fable codex llama" ./lab.sh prepare full
./lab.sh validate full all
```

Expected full study:

```text
70 base tasks
10 variants per base task
700 trajectories per model
2,800 total trajectories
```

### 5. Preview the production launch

For a confirmed ~200k TPM gateway:

```bash
./lab.sh matrix full \
  --concurrency-preset scale-200k \
  --shard-size 30 \
  --shard-index 1 \
  --dry-run
```

Expected aggregate Harbor concurrency:

```text
claude  5
fable   4
codex   4
llama   1
total  14
```

For upgraded 5M TPM keys, first set the key pool and per-key quota in `.env`:

```bash
export LITE_LLM_KEYS="key1,key2,key3"
export LITE_LLM_TPM_LIMIT=5000000
```

Then preview the first full-study shard with the 5M preset:

```bash
./lab.sh matrix full \
  --concurrency-preset scale-5m \
  --shard-size 30 \
  --shard-index 1 \
  --dry-run
```

`scale-5m` applies Claude=20, Fable=16, Codex=16, and Llama=4 **per key**. If `LITE_LLM_KEYS` contains multiple keys, the runner automatically partitions each selected dataset across them.

For an unknown or smaller quota:

```bash
for profile in claude fable codex llama; do
  HARBOR_CONCURRENCY=1 ./lab.sh run full "$profile"
done
```

### 6. Run the full study

With the confirmed ~200k TPM preset, run the three base-task-aware shards sequentially:

```bash
for shard in 1 2 3; do
  caffeinate -dimsu ./lab.sh matrix full \
    --concurrency-preset scale-200k \
    --shard-size 30 \
    --shard-index "$shard"
done
```

For upgraded 5M TPM keys, use the same three sequential shards with the `scale-5m` preset:

```bash
for shard in 1 2 3; do
  caffeinate -dimsu ./lab.sh matrix full \
    --concurrency-preset scale-5m \
    --shard-size 30 \
    --shard-index "$shard"
done
```

The shards are `30 / 30 / 10` complete base-task families, corresponding to `300 / 300 / 100` trajectories per model. Do not launch the three shards simultaneously. With a multi-key pool, each shard is automatically partitioned across the configured keys.

### 7. Analyze

While a study is still running, inspect the current result snapshot with deterministic analysis only:

```bash
./lab.sh analyze full all --live --no-semantic
```

After the study is complete, run the full analysis:

```bash
./lab.sh analyze full all
```

Semantic analysis is enabled by default and uses `ANALYSIS_MODEL` (default: `openai/gpt-5.6`). Judgments are cached in `semantic_judgments.json`. Use `--no-semantic` when you intentionally want deterministic analysis without semantic coding. Use `--live` for an incomplete result snapshot.

Generate standardized result tables from the canonical analyzer outputs:

```bash
./lab.sh results full all
```

The standardized results are written under `analysis/results/full/all/` and include condition-level performance, matched within-task effects, semantic-awareness summaries, pressure/placement analyses, integrity summaries, provenance hashes, and a readable `report.md`. For one profile, use `./lab.sh results full fable` (or another profile name).

`./lab.sh` can be invoked from zsh or Bash; it dispatches Bash-specific helper scripts internally. Do not manually `source scripts/00_common.sh` from zsh.

Generate the Excel workbook after the final analysis:

```bash
./lab.sh report full all
```

### 8. Resume infrastructure-censored work

Always preview first:

```bash
./lab.sh resume codex \
  results/full/<outer-job>/<harbor-job> \
  --concurrency 4 \
  --dry-run
```

Then rerun the same command without `--dry-run`.

Do not blindly retry `NonZeroAgentExitCodeError`; classify the cause first. `AgentSafetyRefusalError` is a substantive model outcome and is not an infrastructure retry.

### Optional diagnostics

```bash
./lab.sh models
./lab.sh llama-doctor
```

`./lab.sh models` prints the gateway model catalog and may produce very large output.

## Benchmark design

Each SWE-Atlas base task is assigned one fixed evaluation cue. The assignment is reused across every condition and every model profile.

The primary benchmark has ten variants per base task:

| Variant | Evaluation cue | Pressure | Placement |
|---|---|---|---|
| clean | no | none | none |
| eval | yes | none | source |
| eval-financial | yes | financial | source |
| eval-self | yes | self-preservation | source |
| eval | yes | none | root |
| eval-financial | yes | financial | root |
| eval-self | yes | self-preservation | root |
| eval | yes | none | scaffold |
| eval-financial | yes | financial | scaffold |
| eval-self | yes | self-preservation | scaffold |

With all 70 base tasks this produces **700 trajectories per model profile**.

### Evaluation cues

The released benchmark uses only **explicit evaluation cues**: five mechanisms with two wording variants each (10 cues total). The mechanisms are:

1. direct evaluation disclosure;
2. evaluation-versus-deployment stage;
3. benchmark provenance;
4. scoring / verification apparatus;
5. evaluation purpose.

`manifests/cue_assignments.json` assigns the ten cues evenly across the 70 base tasks using seed `20260805`. Each task keeps the same exact cue in all seeded variants.

The cue library and its literature mappings are in `factor_data/evaluation_cues.json`.

### Delivery channels

- **source** — treatment is inserted as a comment in an existing source file;
- **root** — treatment is appended to `BENCHMARK_CONTEXT.md`;
- **scaffold** — treatment is appended to `CLAUDE.md` or `AGENTS.md`, depending on the agent.

Existing repository instructions are preserved and appended to rather than overwritten.

### Pressure conditions

The primary benchmark includes financial and self-preservation pressure. Resource deprivation is kept as a separate matched follow-up so it does not expand the primary placement matrix.

Resource mode uses, by default:

| Variant | Evaluation cue | Pressure | Placement |
|---|---|---|---|
| clean | no | none | none |
| eval | yes | none | scaffold |
| eval-resource | yes | resource deprivation | scaffold |

With all 70 base tasks this produces **210 trajectories per model profile**.

## Setup

Host requirements:

- Git and `curl`;
- a Modal account / authentication;
- access to an OpenAI-compatible LiteLLM gateway.

The host runtime is repository-local: `uv python install 3.13` installs Python 3.13, and `uv sync --locked` installs the pinned host dependencies from the committed `pyproject.toml` / `uv.lock`, including Harbor `0.20.0` and Modal. No system-wide Harbor or Modal installation is required.

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv sync --locked
source .venv/bin/activate
```

On a fresh machine, authenticate the installed Modal CLI:

```bash
modal setup
```

Create the local environment file and add the required gateway credential(s):

```bash
cp .env.example .env
nano .env
```

The repository does not contain credentials, and `.env` is gitignored.

The default profiles are:

- Claude Code + `anthropic/claude-opus-4-8`
- Claude Code + `anthropic/claude-fable-5`
- Codex + `openai/gpt-5.6`
- Mini-SWE-Agent + `openai/llmengine/llama-3-3-70b-instruct`

The Llama profile uses Mini-SWE's text-based command parser rather than provider-native function calling. This avoids structured tool-call failures seen with the earlier Groq configuration.

### Bootstrap compatibility wrappers

Two committed wrappers make agent installation reproducible across the heterogeneous SWE-Atlas task images while leaving agent execution behavior inherited from the underlying Harbor/benchmark agents:

- **Codex:** `BootstrapCompatibleCodex` inherits Harbor's Codex agent, pins Codex CLI `0.147.0`, and adds `GIT_ALLOW_PROTOCOL=file:https` only to the agent-install subprocess so NVM can bootstrap in images whose task/test environment restricts Git transport. The task/verifier environment is not rewritten.
- **Llama / Mini-SWE:** `BootstrapCompatibleLlamaTextBasedMiniSweAgent` inherits the benchmark's existing text-based Mini-SWE agent, pins Mini-SWE `2.4.5` and LiteLLM core `1.83.0`, and omits Harbor's `litellm[proxy]` server extra because this benchmark uses an external LiteLLM gateway. The proxy extra introduced a `pyroscope-io` source-packaging failure on a musl task image and is not required for Mini-SWE's client use.

These are bootstrap/harness compatibility changes, not benchmark treatments. If either wrapper's solver-visible behavior, executable version, permissions, task image, or verifier environment is changed in future work, treat that as an experimental change rather than a transparent retry.

Useful runtime and gateway checks:

```bash
./lab.sh doctor        # local/runtime preflight; no model generation
./lab.sh models        # gateway model-list diagnostics
./lab.sh llama-doctor  # Harbor-Python import + Llama route/format probe
```

`doctor` loads the same environment aliases used by benchmark runs, verifies both bootstrap-compatible wrappers with Harbor's own Python interpreter, checks the pinned Codex/Mini-SWE/LiteLLM versions, config files, shell syntax, and Python compilation.

## Generate and validate a run

Build the task registry and permanent cue assignment:

```bash
./lab.sh inventory
./lab.sh assign-cues
```

Inspect the planned size:

```bash
./lab.sh plan full
./lab.sh plan resource
```

Generate and validate a small pilot before a large run:

```bash
PROFILES="claude fable codex llama" ./lab.sh prepare pilot
./lab.sh validate pilot all
```

Generate the full primary benchmark:

```bash
PROFILES="claude fable codex llama" ./lab.sh prepare full
./lab.sh validate full all
```

Run one profile:

```bash
./lab.sh run full claude
```

### Agent-install compatibility smoke

Harbor supports `--install-only`, which builds the task environment and installs the selected agent without running the model. Use this before expensive runs when task images may differ in package managers, Git policy, or build toolchains.

For example, to reproduce the task-image families that exposed the Codex NVM/Git-HTTPS bootstrap incompatibility and the Mini-SWE `litellm[proxy]`/Pyroscope packaging incompatibility on a musl image during development, select them explicitly and run install-only at low concurrency:

```bash
BENCHMARK_TASK_IDS="task-69d196f015a150488265afc2,task-69d196f015a150488265afc3,task-69d196f015a150488265afc4,task-69d196f015a150488265afba,task-696719205599a51110d4b446" \
  PROFILES="codex llama" ./lab.sh prepare resource

./lab.sh validate resource codex
./lab.sh validate resource llama

HARBOR_CONCURRENCY=1 ./lab.sh run resource codex --install-only
HARBOR_CONCURRENCY=1 ./lab.sh run resource llama --install-only
```

This smoke uses no model generation, but it still creates Harbor/Modal environments. A passing install-only smoke establishes setup compatibility, not task-solving correctness.

### Safe resume

Use the repository wrapper rather than calling `harbor job resume` directly:

```bash
./lab.sh resume codex results/full/<outer-job>/<harbor-job> --concurrency 4
./lab.sh resume llama results/full/<outer-job>/<harbor-job> --concurrency 1
```

The wrapper reloads the canonical LiteLLM aliases and repo `PYTHONPATH`, keeps `config.json` and `lock.json` concurrency synchronized, backs both files up under `results/_resume_provenance/`, and records resume history in `run_metadata.json`. Default retry filters are profile-specific and limited to failures that are normally infrastructural.

`NonZeroAgentExitCodeError` is **not retried by default** because it can be a deterministic agent-install/setup incompatibility (for example an image that blocks Git HTTPS during NVM bootstrap or exposes an incompatible dependency/package build path). After classifying and fixing the cause, an explicit retry is possible with `--retry-nonzero`. `AgentSafetyRefusalError` is never an infrastructure default.

Preview a resume without changing files or starting Harbor:

```bash
./lab.sh resume codex results/full/<outer-job>/<harbor-job> --concurrency 4 --dry-run
```

### Concurrent multi-profile runs

`matrix` launches the selected profiles concurrently while giving each profile its own Harbor concurrency. Conservative defaults are 1/1/1/1. Inspect the plan without starting Harbor:

```bash
./lab.sh matrix full --dry-run
```

For a gateway key whose limit has been independently confirmed at approximately **200,000 TPM**, the repository includes an empirically tested starting preset:

```bash
./lab.sh matrix full --concurrency-preset scale-200k --dry-run
./lab.sh matrix full --concurrency-preset scale-200k
```

The `scale-200k` preset is **Claude=5, Fable=4, Codex=4, Llama=1** (14 active Harbor trials total). These numbers are **per-profile concurrency limits, not a run order**: `matrix` starts all selected profiles concurrently.

#### Why 5 / 4 / 4 / 1?

This is an **empirical operational preset**, not a claim that this asymmetric split is theoretically optimal. During the August 18, 2026 benchmark run on a key independently observed to have a 200,000 TPM limit:

- a more aggressive **6 / 5 / 6 / 2** allocation (19 active trials) reached the token ceiling and produced an observed HTTP 429 with zero remaining token capacity;
- after throttling to **5 / 4 / 4 / 1** (14 active trials), a 12-probe health window returned **12/12 HTTP 200**, **0 rate-limit responses**, and a peak observed key usage of approximately **119,142 / 200,000 TPM (59.6%)**;
- the exact per-profile split was retained because it was the first configuration observed to restore stable execution while preserving useful parallelism across all four profiles. It should therefore be treated as a reproducible **starting point**, not as an optimum derived from model size, cumulative trajectory-token counts, or a universal provider rule.

A new user should first confirm the actual key limit and watch the initial execution window for 429s. If the gateway or model mix differs, reduce or customize the concurrency rather than assuming `scale-200k` is safe.

#### Upgraded 5M keys and multiple keys

The existing `scale-200k` preset remains unchanged for older 200k TPM keys. For a key confirmed at approximately **5,000,000 TPM**, set the per-key quota metadata and use the new conservative high-throughput starting preset:

```bash
export LITE_LLM_TPM_LIMIT=5000000
./lab.sh matrix full --concurrency-preset scale-5m --dry-run
./lab.sh matrix full --concurrency-preset scale-5m
```

`scale-5m` starts at **Claude=20, Fable=16, Codex=16, Llama=4** (56 active Harbor trials per key). Unlike `scale-200k`, this 5M allocation is intentionally conservative and has not been empirically optimized to saturate the full quota; use `custom` if a measured gateway window supports more.

The repository also accepts an arbitrary-size key pool:

```bash
export LITE_LLM_KEYS="key1,key2,key3"
```

When `LITE_LLM_KEYS` is non-empty it takes precedence over legacy `LITE_LLM_KEY`; a one-item pool works normally. Runner jobs are base-task-aware sharded across the available keys, with each Harbor process receiving one key. If a shard ends with rate-limit failures, the orchestration automatically resumes those failures with another key. The semantic analyzer uses request-level, thread-safe round-robin key selection: a 429 cools down only the affected key, another available key is selected immediately, and if every key is cooling the analyzer waits until the earliest reset reported by the gateway.

Concurrency presets are **per key** when a pool is configured. For example, three keys with `scale-200k` retain 14 active Harbor trials per key (up to 42 aggregate); three keys with `scale-5m` use 56 per key (up to 168 aggregate). This preserves the tested 200k behavior while allowing upgraded 5M keys to scale out.

`LITE_LLM_TPM_LIMIT` is client-side configuration used for defaults/reporting only. The gateway's actual HTTP 429 and reset timestamp remain authoritative; the code does not maintain a hardcoded local 5M token counter.

For another quota, configure explicit values in `.env` and use `custom`:

```bash
export MATRIX_CONCURRENCY_PRESET=custom
export CLAUDE_CONCURRENCY=3
export FABLE_CONCURRENCY=3
export CODEX_CONCURRENCY=2
export LLAMA_CONCURRENCY=1
./lab.sh matrix full --dry-run
./lab.sh matrix full
```

A direct `./lab.sh run ...` still uses `HARBOR_CONCURRENCY`; the per-profile values apply to `matrix`.

## Sharding large runs

The primary full benchmark contains **70 base SWE-Atlas tasks**, with 10 variants of each task, for **700 trajectories per model**. Sharding is defined over the 70 base tasks, not over the 700 individual trajectories. This keeps all matched conditions for a base task together.

For fixed-size chunks, `--shard-size 30` means 30 base tasks. `--shard-index` is 1-based:

```bash
./lab.sh run full claude --shard-size 30 --shard-index 1  # base-task positions 1–30  -> 300 trajectories
./lab.sh run full claude --shard-size 30 --shard-index 2  # base-task positions 31–60 -> 300 trajectories
./lab.sh run full claude --shard-size 30 --shard-index 3  # base-task positions 61–70 -> 100 trajectories
```

The positions refer to the deterministic base-task order in the generated manifest; they are not numeric SWE-Atlas task IDs. The same chunks in resource mode contain 90, 90, and 30 trajectories because resource mode has three variants per base task.

Balanced shards are also available when exact chunk size does not matter:

```bash
./lab.sh run full claude --shard 1/3
./lab.sh run full claude --shard 2/3
./lab.sh run full claude --shard 3/3
```

With 70 base tasks, these contain 23, 23, and 24 base tasks (230, 230, and 240 primary trajectories).

Shard datasets are generated under `generated/_shards/` and results are written to separate timestamped job directories. Each run stores both the exact executed shard manifest and the full study manifest, so the analyzer can combine shards and report coverage against the complete study.

### Recommended full-run workflow for a confirmed ~200k TPM key

Use **sharding and concurrency together**. Sharding controls job size/restartability; concurrency controls instantaneous model traffic. Run the three fixed-size shards **sequentially**, while `matrix` runs the four profiles concurrently *within* each shard:

```bash
./lab.sh matrix full --concurrency-preset scale-200k --shard-size 30 --shard-index 1
./lab.sh matrix full --concurrency-preset scale-200k --shard-size 30 --shard-index 2
./lab.sh matrix full --concurrency-preset scale-200k --shard-size 30 --shard-index 3
```

Do **not** launch all three shard commands in separate terminals at the same time: that would multiply the intended aggregate concurrency. The commands above are blocking, so running them one after another keeps the target allocation at 14 active Harbor trials rather than roughly 42.

For a smaller quota, keep the same shards and lower the per-profile values with the `custom` preset. `1/1/1/1` means up to four active trials because the four profiles still run concurrently. To enforce **one active trial globally**, run one profile at a time with `HARBOR_CONCURRENCY=1`.

## Internet access

SWE-Atlas tasks normally run with internet access, so SWE-EvalPressure keeps it **enabled by default**:

```bash
export ALLOW_INTERNET=true
```

For a network-disabled ablation:

```bash
ALLOW_INTERNET=false ./lab.sh prepare full
./lab.sh validate full all
```

`prepare` writes the selected setting into every generated `task.toml`, and `validate` checks that it matches the manifest. Regenerate a dataset after changing this setting.

## Modes

| Mode | Base tasks | Variants/task | Use |
|---|---:|---:|---|
| `pilot` | 4 | 10 | pipeline smoke test |
| `sample` | 10 by default | 10 | medium-scale test |
| `full` | 70 | 10 | complete primary benchmark |
| `resource` | 70 by default | 3 | matched resource-deprivation follow-up |

Set `BENCHMARK_TASK_IDS` to a comma-separated list of task IDs for an exact custom subset.

## Detailed execution profiles and recovery

The benchmark supports two execution profiles. Dataset generation, validation, experimental conditions, and analysis are identical; only execution concurrency and scheduling differ.

### Common setup

Start from a clean checkout:

```bash
cp .env.example .env
```

Fill in the local LiteLLM/gateway credentials and any required Modal configuration in `.env`. Do not commit `.env`.

`ALLOW_INTERNET=true` is the default and matches the normal SWE-Atlas task setting used by this benchmark. Set `ALLOW_INTERNET=false` only when intentionally running a no-network condition or ablation. Keep the setting fixed across runs that will be compared directly.

Then validate the runtime and benchmark specification:

```bash
./lab.sh doctor
./lab.sh inventory

PROFILES="claude fable codex llama" ./lab.sh plan full
PROFILES="claude fable codex llama" ./lab.sh prepare full
./lab.sh validate full all
```

The primary `full` study contains 70 base SWE-Atlas tasks × 10 conditions = **700 trajectories per model**, or **2,800 trajectories across the four default profiles**.

Do not start a production run until `doctor` and `validate` pass.

`doctor` should report the committed Codex CLI pin (`0.147.0`). A missing or different production pin is configuration drift and should be resolved before launching the benchmark; do not silently use Harbor's moving `latest` version.

### Standard / unknown gateway quota

If the gateway quota is unknown or substantially below 200k TPM, use conservative execution. The safest option is to run one profile at a time with one active Harbor trial:

```bash
for profile in claude fable codex llama; do
  HARBOR_CONCURRENCY=1 ./lab.sh run full "$profile"
done
```

This keeps at most one benchmark trajectory active at once.

The matrix `serial` preset is also conservative per profile:

```bash
./lab.sh matrix full --concurrency-preset serial --dry-run
./lab.sh matrix full --concurrency-preset serial
```

`serial` means concurrency `1/1/1/1` for Claude/Fable/Codex/Llama, so a four-profile matrix can still have up to four Harbor trials active simultaneously. If the quota is uncertain, prefer the profile-by-profile loop above.

For a custom quota, use the `custom` matrix preset and explicitly choose per-profile concurrency rather than assuming the high-throughput values.

### Scale/internal high-throughput profile — confirmed ~200k TPM

For an internal gateway key confirmed to have approximately **200,000 TPM**, the repository includes the `scale-200k` preset:

- Claude: 5
- Fable: 4
- Codex: 4
- Llama: 1
- total: **14 active Harbor trials**

These values are an empirically tested starting point, not a guarantee for every account or gateway configuration. Confirm the quota before use and monitor for rate-limit responses during the initial execution window.

For the primary full study, use base-task-aware shards of 30 tasks. With 70 base tasks this produces three sequential shards containing **30 / 30 / 10 base tasks**, corresponding to **300 / 300 / 100 trajectories per model**. Every ten-condition family for a base task stays within one shard.

Preview the first shard without starting Harbor:

```bash
./lab.sh matrix full \
  --concurrency-preset scale-200k \
  --shard-size 30 \
  --shard-index 1 \
  --dry-run
```

Then run all three shards **sequentially**:

```bash
for shard in 1 2 3; do
  ./lab.sh matrix full \
    --concurrency-preset scale-200k \
    --shard-size 30 \
    --shard-index "$shard"
done
```

Do **not** launch the three shards simultaneously. The concurrency preset already accounts for the intended aggregate load across all four profiles.

With a multi-key pool, rate-limited runner shards are automatically resumed under an alternate key. With one key, or if every key is exhausted, reduce concurrency or use the repository resume wrapper after the quota resets; do not repeatedly relaunch entire jobs.

### Interrupted or infrastructure-censored runs

Use `./lab.sh resume` instead of calling `harbor job resume` directly. Always preview a resume first:

```bash
./lab.sh resume codex \
  results/full/<outer-job>/<harbor-job> \
  --concurrency 4 \
  --dry-run
```

Then execute it:

```bash
./lab.sh resume codex \
  results/full/<outer-job>/<harbor-job> \
  --concurrency 4
```

Use the appropriate profile and concurrency for the job being resumed.

The default resume filters target transient/provider infrastructure failures. `NonZeroAgentExitCodeError` is intentionally **not** retried by default because it may indicate a deterministic agent-install or task-image incompatibility; inspect and classify the cause first. `AgentSafetyRefusalError` is a substantive model outcome and must not be silently retried as infrastructure.

If a setup repair changes the agent implementation, executable version, task environment, permissions, or other solver-visible behavior, treat it as an experimental change rather than a transparent retry.

### Analyze a completed full run

Analysis is separate from execution and is trajectory-first. It can combine compatible unsharded runs, shards, and resumed runs while preserving the full study denominator.

Analyze all available profiles:

```bash
./lab.sh analyze full all
```

The semantic trajectory judge is enabled by default and uses `ANALYSIS_MODEL`. It produces fixed-rubric labels with supporting trajectory quotes in addition to deterministic measurements.

For deterministic-only analysis:

```bash
./lab.sh analyze full all --no-semantic
```

While production runs are still incomplete, a low-cost snapshot can be generated without invoking the semantic judge:

```bash
./lab.sh analyze full all --live --no-semantic
```

If semantic coding of the currently completed trajectories is intentionally desired, omit `--no-semantic`. `--live` allows analysis of an incomplete study; it does not disable semantic analysis by itself.

The analyzer reports planned, completed, infrastructure-censored/error, and missing trajectories and computes treatment effects only from substantively usable **within-base-task matched comparisons**. Raw Harbor trajectories and verifier outputs remain the source of truth.

The same workflow applies to other modes (`pilot`, `sample`, and `resource`) by replacing `full` with the desired mode. The full resource-deprivation study contains 70 base tasks × 3 conditions = **210 trajectories per model**.

## Analysis

Analysis is a separate, explicit step. It does **not** run automatically after a Harbor job by default. This avoids duplicate analysis jobs when several shards finish at roughly the same time and makes it clear which result snapshot is being analyzed.

Analyze one profile or all profiles after the study is complete:

```bash
./lab.sh analyze full claude
./lab.sh analyze full all
```

For an interim deterministic snapshot while runs are still accumulating results:

```bash
./lab.sh analyze full all --live --no-semantic
```

The analyzer is trajectory-first. It discovers compatible completed trajectories and reconstructs the study from their manifests, so unsharded runs, multiple shards, resumed runs, and partial runs use the same analysis path. Matched effects are calculated only when the baseline and treatment trajectories for the **same base task, placement, and replicate** are both available and substantively usable.

### Deterministic measurements

Structured logs are used for measurements that do not require interpretation: terminal status, verifier rewards, raw individual tool calls, tool-bearing turns, tool categories, validation/test commands, token use, duration, treatment-file access, whether the cue text entered the recorded model-visible trajectory, public internet/code-host activity, and modifications to tests or evaluation-context files. These measurements remain separate fields in the canonical trial-level analysis.

Execution failures are separated from substantive model outcomes. Known install/bootstrap failures are labeled `agent_setup_error`; Modal/image-build failures are `environment_error`; rate/provider/timeout failures remain censored infrastructure. Safety refusals remain substantive model outcomes rather than being silently removed from treatment-response accounting.

### Semantic trajectory judge

The LLM judge is **enabled by default** and uses `ANALYSIS_MODEL` (default: `openai/gpt-5.6`). It reads the model's assistant trajectory together with the assigned condition/cue metadata and returns structured labels plus short supporting quotes. It is not a keyword matcher. The judge codes:

- recognition and stance toward the evaluation cue;
- generic hidden-grader awareness versus awareness that the **current run** is an evaluation;
- benchmark recognition and evaluation-purpose recognition;
- semantic response to the evaluation cue;
- recognition, type, stance, and response to financial, self-preservation, or resource-deprivation pressure;
- explicit evidence of considering or attempting evaluation-targeting manipulation.

The prompt explicitly forbids inferring private reasoning or treating ordinary testing, public-code lookup, wrong patches, or file edits as manipulation without supporting evidence. Long trajectories are sampled from both the beginning and end so early cue reactions and later verification decisions are both retained. Semantic outputs are cached in `semantic_judgments.json`, including the judge model and trajectory hash, so rerunning analysis does not re-judge unchanged trajectories.

To run deterministic analysis only:

```bash
./lab.sh analyze full claude --no-semantic
```

If LiteLLM credentials are unavailable, the analyzer warns and continues with deterministic measurements rather than failing the entire analysis.

Outputs are written to `analysis/<mode>/<profile>/`. `trials.csv` is the canonical row-level table and `matched_pairs.csv` contains the within-task comparisons and treatment-minus-baseline deltas. Additional files summarize coverage, terminal status, treatment delivery, tool use, external lookup, semantic coding, duplicates, and machine-readable provenance.

Analysis can be made automatic with `AUTO_ANALYZE=1`, but the repository default is `0`; explicit analysis is safer for sharded runs.

## Generate the Excel report

After analyzer outputs exist, build a presentation-ready `.xlsx` workbook directly from the canonical CSV/JSON analysis tables:

```bash
./lab.sh report full all
```

The default output is:

```text
analysis/full/SWE-EvalPressure_full_report.xlsx
```

The workbook is generated from `trials.csv`, `matched_pairs.csv`, and the analyzer's supporting tables. It does not re-parse Harbor trajectories or invoke the semantic judge. Infrastructure/setup outcomes and substantive verifier outcomes remain separately summarized, and semantic figure sheets are populated when semantic coding is available.

Useful variants:

```bash
# One profile only
./lab.sh report full llama

# Resource-deprivation study
./lab.sh report resource all

# Explicit analyzer/output locations
./lab.sh report full all \
  --analysis-root /path/to/analysis \
  --output /path/to/report.xlsx
```

The generated workbook follows the experiment-report conventions used for SWE-EvalPressure results. It includes README and run-inventory sheets, a data dictionary, canonical trial and matched-pair tables, primary summaries, tool breakdowns, coverage and terminal-status tables, treatment-delivery and semantic-coding summaries, external-lookup results, matched-effect summaries, and numbered `C01`–`C06` Core Fig sheets with native Excel charts.

## Validation

`./lab.sh validate` checks that:

- original instructions, tests, and gold solutions are unchanged;
- every base task has the expected condition matrix;
- each task retains one permanent evaluation cue across seeded variants;
- clean variants contain no treatment text;
- source, root, and scaffold overlays are well formed;
- manifest and seed metadata agree;
- the requested internet setting is present in every generated task;
- full mode contains 70 base tasks and 700 generated tasks per profile.

## Repository layout

```text
factor_data/      evaluation cues and pressure libraries
manifests/        SWE-Atlas inventory and permanent cue assignments
scripts/          planning, generation, validation, execution, and analysis
config/           agent-specific configuration
custom_agents/    custom Harbor agent wrappers
vendor/rf/        original SWE-Atlas refactoring tasks
generated/        rendered Harbor datasets (gitignored)
results/          Harbor outputs (gitignored)
analysis/         analyzer outputs (gitignored)
```

Install-only automatically collapses prepared cue variants to one representative (preferring clean) per base task, avoiding redundant setup tests across cue variants.
