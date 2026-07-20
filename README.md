# Codebase Refactor Agent (Legacy Cleaner)

An AI agent that connects to a GitHub repository, analyzes code for "code
smells", creates a refactoring plan, applies changes as diffs in a new
branch, verifies them, and - once a human reviews and approves a diff -
opens a Pull Request.

> **This fork has been through two hardening passes.** The first fixed
> shell injection, unsandboxed execution of untrusted repo code, token
> leakage, and a fully automatic commit/push/PR flow with no review step -
> see [Security model](#security-model). The second replaced whole-file
> rewrites with diff-based editing, reconciled a split state model,
> added cost/scope controls, and added a run log + optional checkpointing -
> see [What's new in this pass](#whats-new-in-this-pass).

## Features

- **Perception**: Reads code from the file system
- **Reasoning/Planning**: Creates multi-step refactoring plans
- **Tool Use**: Integrates with git, linters (ruff, black), pytest, and the GitHub API
- **Memory**: Tracks processed files to avoid reprocessing them unnecessarily
- **Self-Correction**: Analyzes linter/test errors and retries, up to a bounded retry limit
- **Human-in-the-loop**: Nothing is committed, pushed, or opened as a PR without an explicit review/approval step
- **Diff-based editing**: Changes are applied as unified diffs (`git apply`), not whole-file rewrites
- **Cost-aware**: Shows an upfront LLM-call estimate and supports a `--max-files` cap before any calls are made
- **Auditable**: Every run writes a JSON + Markdown log; interrupted runs can resume from a checkpoint

## Architecture

Built with:
- **Language**: Python 3.10+
- **Agent Framework**: LangGraph (explicit state and cycle control)
- **LLM Interface**: Groq
- **Sandboxing**: Docker, for any command that executes code from the cloned (untrusted) repository
- **State Management**: `src/state/models.py` defines `AgentState`, a single Pydantic model that both documents the state shape *and* is what `StateGraph` actually compiles and runs against (see [below](#whats-new-in-this-pass) - this used to be two divergent definitions).

## Installation

```bash
pip install -r requirements.txt
```

You'll also need [Docker](https://docs.docker.com/get-docker/) installed and
running - by default, linting, formatting, and running the test suite happen
inside a sandboxed container rather than on your host, since those commands
execute code that lives inside whatever repository you point this at.

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `GROQ_API_KEY` (for Groq LLM access)
- `GITHUB_TOKEN` (for cloning private repos, pushing, and PR creation)

`LLM_MODEL` defaults to `openai/gpt-oss-120b`. Groq model availability
changes fairly often - check https://console.groq.com/docs/models before
relying on a specific default long-term.

## Usage

```bash
# Add type hints to a repository (interactive: cost estimate, then a diff to approve)
python main.py --repo https://github.com/user/repo --task "Add type hints"

# Refactor specific files
python main.py --repo https://github.com/user/repo --files src/main.py,src/utils.py --task "Improve code structure"

# Auto-approve everything (for CI use - make sure you trust the pipeline that's calling this)
python main.py --repo https://github.com/user/repo --task "Fix code smells" --yes

# Preview what would happen without cloning or changing anything
python main.py --repo https://github.com/user/repo --task "Fix code smells" --dry-run

# Resume a run that crashed or was interrupted
python main.py --resume a1b2c3d4e5f6 --local-path ./refactor-temp-repo-20260719-101500
```

Flags:
- `--branch-name` - custom branch name (auto-generated otherwise)
- `--local-path` - where to clone to (a temp dir is used and cleaned up otherwise); required alongside `--resume`
- `--keep-temp` - don't delete the local clone when the run finishes
- `--max-retries` - executor↔verifier retry cycles before giving up (default 3)
- `--max-files` - cap on Python files processed per step (default 50; `0` disables the cap)
- `--resume RUN_ID` - resume an interrupted run from its last checkpoint (needs `--local-path`)
- `--no-sandbox` - **dangerous**: run lint/test commands directly on the host instead of in Docker; only use this for repositories you fully trust
- `--yes` / `-y` - skip interactive confirmations (cost estimate, diff approval); use in CI

## What's new in this pass

### Diff-based editing, not whole-file rewrites
The executor now asks the LLM for a unified diff and applies it with
`git apply` (`src/tools/patch_tools.py`), instead of regenerating and
overwriting the entire file. This is cheaper (smaller prompts/responses),
more precise (a human reviewing the final diff sees exactly the intended
change, not noise from an unrelated full-file regeneration), and safer in
two structural ways that whole-file rewrites didn't have:

- `git apply --check` rejects a patch that doesn't cleanly apply (wrong
  context, stale line numbers) *before* anything is written.
- The diff's own file headers are validated against the path we asked the
  model to edit (`diff_targets_path`), so a diff can't silently target a
  different file than the one it was supposed to.
- If a patch applies cleanly but leaves invalid Python, the executor now
  has a genuine rollback: it restores the file to its exact pre-patch
  content (kept in memory), rather than just refusing to write in the
  first place. Either the patch is fully applied and valid, or the file is
  exactly as it was before the attempt - never something in between.

### Single `AgentState` model
`src/state/models.py`'s Pydantic `AgentState` is now what `StateGraph` is
built with (`StateGraph(AgentState)`), replacing the old `TypedDict` that
used to live in `src/nodes/workflow.py` and quietly diverge from the
Pydantic model. Node functions read state via attribute access
(`state.repo_path`) and return plain dicts of the fields they changed;
`messages` and `execution_history` use an `operator.add` reducer so nodes
only need to return their *new* entries, not the whole accumulated list.

**Version note:** LangGraph's exact behavior for Pydantic-schema graphs
(e.g. whether `.invoke()` returns a dict or a model instance) has shifted
across versions and wasn't verifiable against a live install while writing
this. `main.py` and `github_integration_node` both normalize incoming state
with `state if isinstance(state, dict) else state.model_dump()` specifically
so this stays correct either way - if you hit an incompatibility, that's
the first place to check.

### Cost / scope controls
Before any LLM call is made, the CLI shows how many Python files were
found, applies `--max-files` if the repo exceeds it, and prints an upfront
estimate (`estimate_llm_calls` in `src/nodes/workflow.py`) of the maximum
number of LLM calls the run could make - then asks for confirmation unless
`--yes` is passed. `llm_calls_used` tracks the *actual* count through the
run and is shown in the summary and the run log.

### Run log + optional checkpointing
Every run writes `run.json` and `run.md` to `./prologue-runs/<branch>-<id>/`
(`src/tools/run_log.py`) - task, status, files touched, verification
results, retries and LLM calls used, and the PR URL if one was created.
This is written outside the cloned repo so it can never be swept into
`git add -A`. Independently, `build_refactor_graph` accepts an optional
LangGraph checkpointer; if `langgraph-checkpoint-sqlite` is installed and
working, the CLI wires up a SQLite-backed checkpointer automatically and
prints a run ID you can pass to `--resume` if the process crashes mid-run.
**This is best-effort**: if checkpointer construction fails for any reason
(version mismatch, missing package), the tool prints a note and continues
without it - `--resume` just won't be available for that run. The run log
is not affected either way.

## Security model

The first hardening pass focused on four issues:

### 1. Shell injection → fixed
Every subprocess call uses list-form arguments with `shell=False` (the
default). No command is built by interpolating a variable into a shell
string. As defense-in-depth against *argument* injection (a filename or
branch name starting with `-` being read as a flag), branch names are
validated against an allow-list pattern before touching git, and a literal
`--` separator is inserted before path arguments for every tool that
supports it (including diff application - see `patch_tools.py`).

### 2. No sandboxing → fixed (opt-out, not default)
`ruff`, `black`, and `pytest` all execute code that lives inside the cloned
repository. Those commands run inside a disposable, network-isolated,
resource-limited Docker container (`src/tools/sandbox.py`) by default. If
Docker isn't available, the tool refuses to silently fall back to running
that code on the host. Git operations and diff application still run on
the host, since neither executes arbitrary repo-supplied code the way
`pytest` does.

### 3. Token handling → fixed
The GitHub token is never embedded in the clone/push URL. It's injected as
a short-lived HTTP Basic Auth header via `git --config-env=...`
(`src/tools/git_auth.py`), so it never appears in a subprocess's argument
list and the remote URL persisted to `.git/config` is the plain, tokenless
URL. Local clone directories are cleaned up automatically once a run
finishes (`--keep-temp` to opt out).

### 4. Blind auto-commit/push/PR → fixed
The LangGraph workflow ends at an `awaiting_approval` state once
verification passes - it does not flow straight into committing, pushing,
and opening a PR. The CLI shows a diff and requires confirmation (or
`--yes`) before shipping changes.

### Known limitations
- **Python-only.** `find_python_files` only looks for `.py` files, and the
  diff-based editing prompts are Python-specific.
- **Multi-file diffs aren't supported.** Diffs are requested and applied
  one file at a time; a refactor that inherently spans multiple files in
  one atomic change isn't modeled.
- **Checkpointing is best-effort**, per the version note above - verify it
  actually works against your installed LangGraph/checkpoint package
  versions before relying on `--resume` for anything important.
- **Cost estimate is an upper bound, not a prediction.** It assumes every
  file needs every retry attempt; actual usage (`llm_calls_used`) is
  usually lower.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── tools/
│   │   ├── file_tools.py     # File I/O, git plumbing, lint/format/test dispatch
│   │   ├── github_tools.py   # Clone, PR creation, GitHub API
│   │   ├── git_auth.py       # Header-based git HTTPS auth (no token in argv/URL)
│   │   ├── sandbox.py        # Docker-based sandboxed execution
│   │   ├── patch_tools.py    # Diff validation + application via git apply
│   │   └── run_log.py        # JSON + Markdown run log, written outside the repo
│   ├── nodes/           # Agent nodes (planner, executor, verifier, github_integration)
│   └── state/           # Single canonical AgentState (Pydantic)
├── tests/               # Test suite: security + diff-editing + state + run-log regressions
├── main.py              # CLI entry point (clone → estimate/confirm → plan/edit/verify → review → ship → log)
├── requirements.txt
└── .env.example
```

## How It Works

1. **Clone**: repository is cloned via a tokenless URL with header-based auth
2. **Cost check**: files are counted, `--max-files` applied if needed, an LLM-call estimate shown, confirmation required (or `--yes`)
3. **Planner Node**: analyzes the request and breaks it into executable steps
4. **Executor Node**: requests a diff per file, applies it via `git apply`, syntax-checks the result, rolls back on failure
5. **Verifier Node**: runs linters and tests inside the sandbox; loops back to the executor on failure, bounded by `--max-retries`
6. **Human review**: the CLI shows a diff and asks for approval (or requires `--yes`)
7. **GitHub Integration**: only after approval - creates branch, commits, pushes, opens PR
8. **Run log**: written to `./prologue-runs/` regardless of outcome

## License

MIT
