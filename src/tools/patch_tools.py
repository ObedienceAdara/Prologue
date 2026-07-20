"""Diff/patch-based file editing.

Why diffs instead of whole-file rewrites
-----------------------------------------
The previous executor sent the LLM an entire file and overwrote it with
whatever came back. That's expensive (full file in, full file out, on
every attempt) and imprecise: a subtly-wrong or truncated response quietly
replaces content that had nothing to do with the requested change, and the
diff a human reviews later is the whole file rather than the actual edit.

This module has the LLM emit a unified diff instead, and applies it with
`git apply`, which gives two structural safety properties that whole-file
rewrites didn't have:

  * `git apply --check` fails clearly if the patch doesn't cleanly apply
    against the file's current content (wrong context, wrong file, stale
    line numbers) - nothing is written to disk in that case.
  * Because the caller (executor_node) keeps the pre-patch content, a patch
    that *does* apply cleanly but produces invalid syntax can be rolled
    back to that exact prior content - a real undo, not just prevention.

`git apply` only checks that a patch applies textually; it has no idea
whether the resulting file is valid Python. Syntax validation and rollback
therefore stay the caller's responsibility (see executor_node), not this
module's.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .file_tools import ToolResult, is_safe_relpath

_HEADER_PLUS_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_HEADER_MINUS_RE = re.compile(r"^--- a/(.+)$", re.MULTILINE)


def has_hunks(diff_text: str) -> bool:
    """Whether a diff contains any actual changes (vs. just file headers, or nothing)."""
    if not diff_text:
        return False
    return any(line.startswith("@@") for line in diff_text.splitlines())


def diff_targets_path(diff_text: str, expected_relpath: str) -> bool:
    """Confirm a diff's file headers point at exactly the file we asked the
    model to edit.

    This matters specifically *because* editing is now diff-based: the
    model is shown file content that ultimately comes from an untrusted
    repository, and a diff's file headers are themselves just text the
    model chose to emit. Without this check, a crafted file could attempt
    to get the model to produce a diff that edits some other path (e.g. a
    CI config or a hooks file) instead of the file it was actually asked
    to refactor. `/dev/null` is allowed alongside the real path to permit
    new-file/deleted-file diffs.
    """
    targets = set(_HEADER_PLUS_RE.findall(diff_text)) | set(_HEADER_MINUS_RE.findall(diff_text))
    targets.discard("/dev/null")
    if not targets:
        return False
    return targets == {expected_relpath}


def apply_patch(repo_root: str, relative_path: str, diff_text: str, timeout: int = 30) -> ToolResult:
    """Validate and apply a unified diff to a single file inside repo_root.

    Args:
        repo_root: Absolute path to the repository (the diff is applied
            with this as the working directory, so paths in the diff are
            relative to it).
        relative_path: The path the model was asked to edit - the diff's
            headers must match this exactly (see `diff_targets_path`).
        diff_text: The unified diff to apply.
        timeout: Timeout in seconds for each `git apply` invocation.

    Returns:
        ToolResult. On failure, nothing is written - `git apply --check`
        runs first and any failure there aborts before the real apply.
    """
    if not is_safe_relpath(repo_root, relative_path):
        return ToolResult(success=False, output="", error=f"Refusing patch outside repo root: {relative_path}")

    if not diff_targets_path(diff_text, relative_path):
        return ToolResult(
            success=False,
            output="",
            error=f"Diff header does not match the expected file ({relative_path}); refusing to apply.",
        )

    if not has_hunks(diff_text):
        return ToolResult(success=False, output="", error="Diff contains no hunks (no actual changes) to apply.")

    patch_file: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
            f.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
            patch_file = f.name

        check = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check.returncode != 0:
            return ToolResult(
                success=False, output="", error=f"Patch does not apply cleanly: {check.stderr.strip()}"
            )

        applied = subprocess.run(
            ["git", "apply", patch_file],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if applied.returncode != 0:
            return ToolResult(success=False, output="", error=f"git apply failed: {applied.stderr.strip()}")

        return ToolResult(success=True, output=f"Applied patch to {relative_path}")

    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error="git apply timed out")
    except OSError as e:
        return ToolResult(success=False, output="", error=f"Failed to invoke git: {e}")
    finally:
        if patch_file:
            try:
                Path(patch_file).unlink(missing_ok=True)
            except OSError:
                pass
