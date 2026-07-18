# Codebase Refactor Agent (Legacy Cleaner)

An AI agent that connects to a GitHub repository, analyzes code for "code
smells", creates a refactoring plan, applies changes in a new branch, verifies
them, and - once a human reviews and approves a diff - opens a Pull Request.

> **This fork is a security-hardened rewrite.** The original version had
> several serious vulnerabilities (shell injection, unsandboxed execution of
> untrusted repo code, tokens leaking into `.git/config` and process argv,
> and a fully automatic commit/push/PR flow with no review step). See
> [Security model](#security-model) below for what changed and why.

## Features

- **Perception**: Reads code from the file system
- **Reasoning/Planning**: Creates multi-step refactoring plans
- **Tool Use**: Integrates with git, linters (ruff, black), pytest, and the GitHub API
- **Memory**: Tracks processed files to avoid reprocessing them unnecessarily
- **Self-Correction**: Analyzes linter/test errors and retries, up to a bounded retry limit
- **Human-in-the-loop**: Nothing is committed, pushed, or opened as a PR without an explicit review/approval step

## Architecture

Built with:
- **Language**: Python 3.10+
- **Agent Framework**: LangGraph (explicit state and cycle control)
- **LLM Interface**: Groq
- **Sandboxing**: Docker, for any command that executes code from the cloned (untrusted) repository
- **State Management**: Pydantic models are defined in `src/state/models.py`, but note the limitation below - the LangGraph workflow actually runs on a separate `TypedDict`, not this Pydantic model.

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
# Add type hints to a repository (interactive: you'll be shown a diff and asked to approve it)
python main.py --repo https://github.com/user/repo --task "Add type hints"

# Refactor specific files
python main.py --repo https://github.com/user/repo --files src/main.py,src/utils.py --task "Improve code structure"

# Auto-approve the diff (for CI use - make sure you trust the pipeline that's calling this)
python main.py --repo https://github.com/user/repo --task "Fix code smells" --yes

# Preview what would happen without cloning or changing anything
python main.py --repo https://github.com/user/repo --task "Fix code smells" --dry-run
```

Other flags:
- `--branch-name` - custom branch name (auto-generated otherwise)
- `--local-path` - where to clone to (a temp dir is used and cleaned up otherwise)
- `--keep-temp` - don't delete the local clone when the run finishes
- `--max-retries` - executor↔verifier retry cycles before giving up (default 3)
- `--no-sandbox` - **dangerous**: run lint/test commands directly on the host instead of in Docker; only use this for repositories you fully trust

## Security model

This is a rewrite focused on four issues found in the original implementation:

### 1. Shell injection → fixed
Every subprocess call now uses list-form arguments with `shell=False`
(the default). No command is ever built by interpolating a variable into a
shell string. As defense-in-depth against *argument* injection (a filename
or branch name starting with `-` being read as a flag), branch names are
validated against an allow-list pattern before touching git, and a literal
`--` separator is inserted before path arguments for every tool that
supports it.

### 2. No sandboxing → fixed (opt-out, not default)
`ruff`, `black`, and `pytest` all execute code that lives inside the cloned
repository - by design, that's arbitrary code from a repo you may not fully
trust. Those commands now run inside a disposable, network-isolated,
resource-limited Docker container (`src/tools/sandbox.py`) by default. If
Docker isn't available, the tool refuses to silently fall back to running
that code on the host - it stops and asks you to either install Docker or
pass `--no-sandbox` to explicitly accept the risk. Note that git operations
themselves (clone, commit, push) still run on the host, since git itself
isn't executing arbitrary repo-supplied code the way `pytest` does.

### 3. Token handling → fixed
The GitHub token is no longer embedded in the clone/push URL. It's injected
as a short-lived HTTP Basic Auth header via `git --config-env=...`
(`src/tools/git_auth.py`), so it never appears in a subprocess's argument
list (not visible via `ps`/`/proc`) and the remote URL git persists to
`.git/config` afterward is the plain, tokenless URL. Local clone directories
are also cleaned up automatically once a run finishes (use `--keep-temp` to
opt out), whereas the original implementation left them - and any
credentials embedded in them - on disk indefinitely.

### 4. Blind auto-commit/push/PR → fixed
The LangGraph workflow now ends at an `awaiting_approval` state once
verification passes - it no longer flows straight into committing, pushing,
and opening a PR. The CLI shows you a diff of exactly what changed and asks
for confirmation (`click.confirm`) before calling the commit/push/PR step;
`--yes` opts into skipping that prompt for non-interactive/CI use. Generated
code is also syntax-checked (`ast.parse`) before being written to disk, so a
malformed LLM response can no longer silently corrupt a file.

### Known limitations not addressed by this pass
- **Two `AgentState` shapes.** `src/state/models.py` defines a Pydantic
  model that the LangGraph workflow doesn't actually use (the graph runs on
  the `TypedDict` in `src/nodes/workflow.py`). Worth reconciling, but it's a
  type-safety/architecture issue rather than a security one.
- **Whole-file rewrites, not diffs.** The executor still sends full file
  contents to the LLM and overwrites the file with the response (now at
  least syntax-checked first). A diff/patch-based editing approach would be
  safer and more token-efficient, but is a larger redesign than this pass covers.
- **Python-only.** `find_python_files` only looks for `.py` files.
- **No cost/scope controls.** A large repo with several retry cycles can
  still trigger a lot of LLM calls; there's no dry-run cost estimate.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── tools/
│   │   ├── file_tools.py     # File I/O, git plumbing, lint/format/test dispatch
│   │   ├── github_tools.py   # Clone, PR creation, GitHub API
│   │   ├── git_auth.py       # Header-based git HTTPS auth (no token in argv/URL)
│   │   └── sandbox.py        # Docker-based sandboxed execution
│   ├── nodes/           # Agent nodes (planner, executor, verifier, github_integration)
│   └── state/           # State models
├── tests/               # Test suite, including security regression tests
├── main.py              # CLI entry point (clone → plan/edit/verify → review → ship)
├── requirements.txt
└── .env.example
```

## How It Works

1. **Clone**: repository is cloned via a tokenless URL with header-based auth
2. **Planner Node**: analyzes the request and breaks it into executable steps
3. **Executor Node**: applies code changes, syntax-checks them, refuses to write outside the repo root
4. **Verifier Node**: runs linters and tests inside the sandbox; loops back to the executor on failure, bounded by `--max-retries`
5. **Human review**: the CLI shows a diff and asks for approval (or requires `--yes`)
6. **GitHub Integration**: only after approval - creates branch, commits, pushes, opens PR

## License

MIT
