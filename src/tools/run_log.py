"""Persist a record of what an agent run did.

This is a plain, human- and machine-readable summary written once at the
end of a run (or from an exception handler on failure) - it's independent
of, and complements, LangGraph checkpointing (see `build_refactor_graph`'s
`checkpointer` argument in src/nodes/workflow.py), which lets an
interrupted run resume mid-graph. This module is for after-the-fact review:
what happened, what changed, what verification found, and how much it cost.

It intentionally writes outside the cloned repository (never inside
`repo_path`) so it can't accidentally get swept into `git add -A` and
committed alongside the actual code changes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_tools import ToolResult, redact_secrets


def _clean(value: Any, secrets: List[str]):
    """Recursively redact secrets from strings, leave other types alone."""
    if isinstance(value, str):
        return redact_secrets(value, secrets)
    if isinstance(value, dict):
        return {k: _clean(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v, secrets) for v in value]
    return value


def write_run_log(state: Dict[str, Any], output_dir: str, secrets: Optional[List[str]] = None) -> ToolResult:
    """Write `run.json` and `run.md` summarizing an agent run.

    Args:
        state: Final (or in-progress, if called from an error handler)
            agent state, as a plain dict.
        output_dir: Directory to write into. Should be outside the cloned
            repository - see module docstring.
        secrets: Raw secret strings (tokens, API keys) to scrub from every
            string field before writing to disk. Defense-in-depth: nothing
            in this codebase currently stores a token in state, but any
            error message that happens to echo one back is caught here too.

    Returns:
        ToolResult with the paths written on success.
    """
    secrets = secrets or []
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now().isoformat(),
            "repo_url": state.get("repo_url"),
            "branch_name": state.get("branch_name"),
            "task_description": state.get("task_description"),
            "overall_status": state.get("overall_status"),
            "error_message": state.get("error_message"),
            "processed_files": state.get("processed_files", []),
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries"),
            "llm_calls_used": state.get("llm_calls_used", 0),
            "linter_errors": state.get("linter_errors", []),
            "test_failures": state.get("test_failures", []),
            "execution_history": state.get("execution_history", []),
            "pr_url": state.get("pr_url"),
        }
        record = _clean(record, secrets)

        json_path = out / "run.json"
        json_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

        md_path = out / "run.md"
        md_path.write_text(_render_markdown(record), encoding="utf-8")

        return ToolResult(success=True, output=f"Wrote run log to {json_path} and {md_path}")

    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to write run log: {e}")


def _render_markdown(record: Dict[str, Any]) -> str:
    lines = [
        f"# Refactor run - {record['timestamp']}",
        "",
        f"**Repo:** {record['repo_url']}  ",
        f"**Branch:** {record['branch_name']}  ",
        f"**Task:** {record['task_description']}  ",
        f"**Status:** {record['overall_status']}  ",
        f"**LLM calls used:** {record['llm_calls_used']}  ",
        f"**Retries used:** {record['retry_count']}/{record['max_retries']}  ",
        "",
    ]

    if record.get("error_message"):
        lines += [f"**Error:** {record['error_message']}", ""]
    if record.get("pr_url"):
        lines += [f"**Pull request:** {record['pr_url']}", ""]

    lines.append("## Files processed")
    if record["processed_files"]:
        lines += [f"- `{f}`" for f in record["processed_files"]]
    else:
        lines.append("_None_")
    lines.append("")

    lines.append("## Execution history")
    if record["execution_history"]:
        for entry in record["execution_history"]:
            icon = "✅" if entry.get("success") else "❌"
            detail = entry.get("error") or entry.get("action") or ""
            file_path = entry.get("file_path", "")
            lines.append(f"- {icon} `{file_path}` - {detail}".strip())
    else:
        lines.append("_None_")
    lines.append("")

    if record["linter_errors"]:
        lines.append("## Linter issues")
        lines += [f"- {e}" for e in record["linter_errors"]]
        lines.append("")

    if record["test_failures"]:
        lines.append("## Test failures")
        lines += [f"- {e}" for e in record["test_failures"]]
        lines.append("")

    return "\n".join(lines)
