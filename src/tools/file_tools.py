"""Tool implementations for the Codebase Refactor Agent.

Security notes
---------------
Every subprocess call in this module uses **list-form arguments** and
`shell=False` (the `subprocess` default) - never a shell string. This was
the single biggest vulnerability in the original implementation: building
shell command strings out of repo-controlled data (file paths, branch names,
commit messages) meant a maliciously named file in a cloned repository could
execute arbitrary shell commands on the host.

List-form arguments alone don't fully close the door, though: a filename or
branch name that *starts with a dash* can still be interpreted by the
downstream tool as a flag rather than a positional argument ("argument
injection"). To guard against that, this module:
  * validates branch names against a conservative allow-list pattern before
    they touch git at all, and
  * inserts a literal `--` separator before path/positional arguments for
    every tool that supports it, so even a file named `--upload-pack=x` is
    always treated as a filename.

`run_linter`, `run_formatter`, and `run_tests` execute code that lives
*inside the cloned repository* (pytest runs arbitrary test files; linter
plugins/configs can execute code too). By default those three functions
delegate to `sandbox.run_in_sandbox`, which runs the command inside a
network-isolated, resource-limited Docker container rather than directly on
the host. Sandbox use can be explicitly disabled by the caller, but that is
an opt-in risk acceptance, not the default.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

from . import sandbox as _sandbox


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Secret redaction - defense in depth. Tokens should never reach argv or
# subprocess output at all (see git_auth.py), but if a secret ever leaks
# into an error message (e.g. a library's exception text), scrub it before
# it's logged, stored in agent state, or shown to the user.
# ---------------------------------------------------------------------------

def redact_secrets(text: str, secrets: List[str]) -> str:
    """Replace any occurrence of each secret string in `text` with a redaction marker."""
    if not text:
        return text
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "***REDACTED***")
    return redacted


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Conservative subset of valid git ref names: must not start with '-', '.',
# or '/'; no whitespace, no shell/glob metacharacters, no consecutive dots,
# no trailing '.lock'. This is stricter than git actually requires - the
# goal is to reject anything that could be misread as a flag or that looks
# unusual, not to support every legal ref name.
_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


def is_valid_branch_name(name: str) -> bool:
    """Return True if `name` is safe to pass as a git branch name argument."""
    if not name or not _BRANCH_NAME_RE.match(name):
        return False
    if name.startswith("-"):
        return False
    if ".." in name or name.endswith(".") or name.endswith(".lock"):
        return False
    if name.endswith("/") or "//" in name:
        return False
    if any(c in name for c in ("~", "^", ":", "?", "*", "[", "\\", " ")):
        return False
    return True


def is_safe_relpath(base_dir: str, candidate_path: str) -> bool:
    """Return True if `candidate_path` resolves to a location inside `base_dir`.

    Guards against path traversal (e.g. a plan step or repo listing that
    resolves to `../../etc/passwd`) before the path is read, written, or
    passed to a subprocess.
    """
    try:
        base = Path(base_dir).resolve()
        target = Path(candidate_path)
        if not target.is_absolute():
            target = (base / target).resolve()
        else:
            target = target.resolve()
        return target == base or base in target.parents
    except (OSError, ValueError):
        return False


def _looks_like_flag(value: str) -> bool:
    """Heuristic: would a CLI tool likely interpret this as an option instead of a path?"""
    return value.startswith("-")


# ---------------------------------------------------------------------------
# Basic file I/O
# ---------------------------------------------------------------------------

def read_file(file_path: str) -> ToolResult:
    """Read the contents of a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        return ToolResult(success=True, output=content)
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Error reading file: {str(e)}")


def write_file(file_path: str, content: str) -> ToolResult:
    """Write content to a file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(success=True, output=f"Successfully wrote to {file_path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Error writing file: {str(e)}")


# ---------------------------------------------------------------------------
# Command execution - list-form only, no shell
# ---------------------------------------------------------------------------

def run_command(
    command: List[str],
    cwd: Optional[str] = None,
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> ToolResult:
    """Run a command directly on the host.

    Args:
        command: Argv list, e.g. ["git", "status"]. Never a shell string -
            this function always runs with shell=False, so shell
            metacharacters in any argument are inert.
        cwd: Working directory for the command.
        timeout: Timeout in seconds.
        env: Extra environment variables to merge on top of the current
            environment (used for injecting short-lived git auth headers
            without exposing them in argv).

    Returns:
        ToolResult with command output or error.
    """
    if not isinstance(command, (list, tuple)) or not command:
        return ToolResult(success=False, output="", error="command must be a non-empty list of arguments")

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    try:
        result = subprocess.run(
            list(command),
            shell=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"

        return ToolResult(
            success=result.returncode == 0,
            output=output,
            error=None if result.returncode == 0 else f"Command failed with exit code {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"Command timed out after {timeout} seconds")
    except FileNotFoundError as e:
        return ToolResult(success=False, output="", error=f"Command not found: {e}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Error executing command: {str(e)}")


# ---------------------------------------------------------------------------
# Linting / formatting / testing - sandboxed by default because these
# execute code that lives inside the (untrusted) cloned repository.
# ---------------------------------------------------------------------------

_LINTER_ARGV = {
    "ruff": lambda rel: ["ruff", "check", "--", rel],
    "black": lambda rel: ["black", "--check", "--", rel],
    "mypy": lambda rel: ["mypy", "--", rel],
}

_FORMATTER_ARGV = {
    "black": lambda rel: ["black", "--", rel],
    "ruff": lambda rel: ["ruff", "format", "--", rel],
}


def _run_repo_command(
    argv_builder,
    file_or_test_path: str,
    repo_root: Optional[str],
    use_sandbox: bool,
    timeout: int,
) -> ToolResult:
    """Shared helper: run a tool against a path inside a repo, sandboxed by default."""
    if repo_root is None:
        return ToolResult(
            success=False,
            output="",
            error="repo_root is required so the path can be safely resolved and (optionally) sandboxed.",
        )

    if not is_safe_relpath(repo_root, file_or_test_path):
        return ToolResult(success=False, output="", error=f"Refusing to operate outside repo root: {file_or_test_path}")

    abs_path = Path(file_or_test_path)
    if not abs_path.is_absolute():
        abs_path = Path(repo_root) / file_or_test_path
    try:
        rel_path = str(abs_path.resolve().relative_to(Path(repo_root).resolve()))
    except ValueError:
        return ToolResult(success=False, output="", error=f"Path is not inside repo root: {file_or_test_path}")

    argv = argv_builder(rel_path if rel_path != "." else ".")

    if use_sandbox:
        try:
            result = _sandbox.run_in_sandbox(argv, repo_root, timeout=timeout)
        except _sandbox.SandboxUnavailableError as e:
            return ToolResult(success=False, output="", error=str(e))
        return ToolResult(success=result.success, output=result.output, error=result.error)

    # Explicit, caller-acknowledged opt-out of sandboxing: run on the host.
    return run_command(argv, cwd=repo_root, timeout=timeout)


def run_linter(
    file_path: str,
    linter: str = "ruff",
    repo_root: Optional[str] = None,
    use_sandbox: bool = True,
    timeout: int = 300,
) -> ToolResult:
    """Run a linter on a file inside the given repo, sandboxed by default."""
    builder = _LINTER_ARGV.get(linter)
    if builder is None:
        return ToolResult(success=False, output="", error=f"Unknown linter: {linter}")
    return _run_repo_command(builder, file_path, repo_root, use_sandbox, timeout)


def run_formatter(
    file_path: str,
    formatter: str = "black",
    repo_root: Optional[str] = None,
    use_sandbox: bool = True,
    timeout: int = 300,
) -> ToolResult:
    """Run a formatter on a file inside the given repo, sandboxed by default."""
    builder = _FORMATTER_ARGV.get(formatter)
    if builder is None:
        return ToolResult(success=False, output="", error=f"Unknown formatter: {formatter}")
    return _run_repo_command(builder, file_path, repo_root, use_sandbox, timeout)


def run_tests(
    test_path: str = ".",
    cwd: Optional[str] = None,
    use_sandbox: bool = True,
    timeout: int = 300,
) -> ToolResult:
    """Run pytest on the specified path inside the repo, sandboxed by default.

    `cwd` doubles as the repo root for sandboxing purposes.
    """
    builder = lambda rel: ["pytest", "--", rel] if rel != "." else ["pytest", "-v"]
    return _run_repo_command(builder, test_path, cwd, use_sandbox, timeout)


def find_python_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Find all Python files in a directory, confined to that directory."""
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", ".git", "node_modules", ".venv", "venv"]

    python_files = []
    dir_path = Path(directory).resolve()

    for file_path in dir_path.glob("**/*.py"):
        if any(exclude in str(file_path) for exclude in exclude_patterns):
            continue
        # Defensive: glob shouldn't escape dir_path, but confirm anyway
        # in case of symlinks pointing outside the repo.
        if not is_safe_relpath(str(dir_path), str(file_path)):
            continue
        python_files.append(str(file_path))

    return sorted(python_files)


# ---------------------------------------------------------------------------
# Git plumbing - runs on the host (git itself is trusted; only the *payload*
# code from the repo needs sandboxing, handled above). All calls are
# list-form with validated inputs.
# ---------------------------------------------------------------------------

def get_git_status(cwd: str) -> ToolResult:
    """Get git status of the repository."""
    return run_command(["git", "status"], cwd=cwd)


def get_git_diff(cwd: str, stat_only: bool = False) -> ToolResult:
    """Get the working-tree diff of uncommitted changes.

    Used to show the human-in-the-loop review before anything is committed
    or pushed - the agent no longer commits/pushes/opens a PR without the
    operator seeing exactly what changed.
    """
    argv = ["git", "diff"]
    if stat_only:
        argv.append("--stat")
    return run_command(argv, cwd=cwd)


def create_git_branch(branch_name: str, cwd: str) -> ToolResult:
    """Create and checkout a new git branch."""
    if not is_valid_branch_name(branch_name):
        return ToolResult(success=False, output="", error=f"Refusing unsafe branch name: {branch_name!r}")

    result = run_command(["git", "checkout", "-b", branch_name], cwd=cwd)
    if not result.success:
        return result

    return ToolResult(success=True, output=f"Created and checked out branch: {branch_name}")


def commit_changes(message: str, cwd: str) -> ToolResult:
    """Stage and commit all changes."""
    stage_result = run_command(["git", "add", "-A"], cwd=cwd)
    if not stage_result.success:
        return stage_result

    # Commit message is passed as a distinct argv element - no quoting
    # required and no risk of it being split or reinterpreted as flags,
    # since argument order after "-m" makes it unambiguous.
    return run_command(["git", "commit", "-m", message], cwd=cwd)


def push_branch(branch_name: str, cwd: str, token: Optional[str] = None) -> ToolResult:
    """Push a branch to the `origin` remote.

    Args:
        branch_name: Branch to push.
        cwd: Repository root.
        token: If provided, authenticates the push via a short-lived HTTP
            Basic Auth header (see git_auth.py) rather than a token embedded
            in the remote URL. Required if `origin` is a plain (tokenless)
            HTTPS URL, which is now always the case after cloning via
            `clone_repository`.
    """
    if not is_valid_branch_name(branch_name):
        return ToolResult(success=False, output="", error=f"Refusing unsafe branch name: {branch_name!r}")

    argv = ["git"]
    env: Dict[str, str] = {}

    if token:
        from .git_auth import build_https_auth
        extra_args, extra_env = build_https_auth(token)
        argv += extra_args
        env.update(extra_env)

    argv += ["push", "-u", "origin", branch_name]

    return run_command(argv, cwd=cwd, env=env or None)
