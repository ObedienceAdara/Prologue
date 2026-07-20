"""LangGraph nodes for the Codebase Refactor Agent.

Changes in this pass
---------------------
1. **Single state model.** The graph now runs on `AgentState` from
   src/state/models.py (a Pydantic model) instead of a separate, divergent
   `TypedDict` defined here. Node functions read state via attribute access
   (`state.repo_path`) and return plain dicts of the fields they changed -
   LangGraph merges those into a new state instance, using the
   `operator.add` reducers declared on `messages` and `execution_history`
   to *append* rather than replace for those two fields specifically.

2. **Diff-based editing.** `executor_node` now asks the LLM for a unified
   diff instead of a full file rewrite, and applies it via
   `src/tools/patch_tools.apply_patch` (`git apply`). This is cheaper
   (smaller prompts/responses), safer (a patch that doesn't cleanly apply
   is rejected before anything is written), and gives a real rollback path:
   if a patch applies but leaves invalid Python, the file is restored to
   its pre-patch content rather than left corrupted.

3. **Cost tracking.** `llm_calls_used` counts every LLM invocation across
   planning and execution (including retries), surfaced in the CLI summary
   and the run log. `estimate_llm_calls` gives an upfront estimate before
   any calls are made, for --max-files / cost-confirmation purposes.

4. **Optional checkpointing.** `build_refactor_graph` accepts a
   `checkpointer` (e.g. a LangGraph SQLite checkpointer) so a run can be
   resumed by thread_id after a crash. This is layered on top of - not a
   replacement for - the run log written by main.py, which is a plain
   human-readable record independent of whether checkpointing is enabled.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from src.state.models import AgentState

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(llm_text: str) -> Optional[str]:
    """Pull code out of an LLM response, handling the common formatting cases."""
    if not llm_text:
        return None

    text = llm_text.strip()

    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    if text.startswith("```") or text.endswith("```"):
        return None

    return text if text else None


def _extract_diff(llm_text: str) -> Optional[str]:
    """Pull a unified diff out of an LLM response.

    Handles: a fenced ```diff / ```patch / plain ``` block anywhere in the
    response (preferred), or - if there's no fence at all - the raw
    (stripped) text, on the assumption the model followed the "diff only,
    no commentary" instruction. A malformed/partial fence is treated as
    unusable rather than guessed at.
    """
    if not llm_text:
        return None

    text = llm_text.strip()

    match = _DIFF_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    if text.startswith("```") or text.endswith("```"):
        return None

    return text if text else None


def _validate_python_syntax(content: str) -> Optional[str]:
    """Return an error message if `content` isn't valid Python, else None."""
    try:
        ast.parse(content)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"


def _get_llm(model_name: str, temperature: float):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model_name, temperature=temperature)


def estimate_llm_calls(num_files: int, max_retries: int) -> int:
    """Upper-bound estimate of LLM calls for a run: one planning call, plus
    each file processed up to (1 + max_retries) times in the worst case
    (an initial attempt plus one attempt per retry cycle)."""
    return 1 + num_files * (1 + max_retries)


def _relpath(file_path: str, repo_path: str) -> str:
    from pathlib import Path
    return str(Path(file_path).resolve().relative_to(Path(repo_path).resolve()))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Planner node that breaks down the refactoring task into steps."""
    import os
    import json
    from dotenv import load_dotenv

    load_dotenv()

    logger.info(f"Planning refactoring task: {state.task_description}")

    model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm = _get_llm(model_name, temperature=0)

    system_prompt = """You are an expert software architect and refactoring specialist.
Your task is to create a detailed, step-by-step plan for refactoring code.

Given a task description, break it down into concrete, executable steps.
Each step should:
1. Be specific and actionable
2. Target specific files or patterns
3. Include the type of refactoring action (add_type_hints, extract_method, rename_variable, etc.)
4. Be ordered logically (dependencies first)

Return your response as a valid JSON array of steps ONLY. No markdown, no explanations.
Each step must have:
- step_id: integer
- description: string
- file_path: string or null
- action: string (one of: add_type_hints, extract_method, rename_variable, simplify_logic, add_docstring, fix_imports, format_code, analyze, refactor, verify)

Example format:
[{"step_id": 1, "description": "Analyze code", "file_path": null, "action": "analyze"}, {"step_id": 2, "description": "Add types", "file_path": "src/main.py", "action": "add_type_hints"}]
"""

    user_prompt = f"""Task: {state.task_description}
Repository: {state.repo_url}
Target files: {state.target_files or 'All Python files'}

Create a detailed refactoring plan as a JSON array."""

    llm_calls = state.llm_calls_used + 1

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        plan_text = response.content.strip()

        if plan_text.startswith("```"):
            lines = plan_text.split("\n")
            start_idx = 0
            for i, line in enumerate(lines):
                if not line.startswith("```"):
                    start_idx = i
                    break
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if not lines[i].startswith("```"):
                    end_idx = i + 1
                    break
            plan_text = "\n".join(lines[start_idx:end_idx])

        steps = []
        try:
            parsed_steps = json.loads(plan_text)
            if isinstance(parsed_steps, list):
                for i, step in enumerate(parsed_steps):
                    if isinstance(step, dict):
                        steps.append({
                            "step_id": step.get("step_id", i + 1),
                            "description": step.get("description", f"Step {i + 1}"),
                            "file_path": step.get("file_path"),
                            "action": step.get("action", "refactor")
                        })
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using fallback plan")

        if not steps:
            steps = [
                {"step_id": 1, "description": f"Analyze codebase for {state.task_description}", "file_path": None, "action": "analyze"},
                {"step_id": 2, "description": f"Apply refactoring: {state.task_description}", "file_path": None, "action": "refactor"},
                {"step_id": 3, "description": "Run linters and formatters", "file_path": None, "action": "verify"},
            ]

        logger.info(f"Created plan with {len(steps)} steps")

        return {
            "plan": steps,
            "current_step_index": 0,
            "overall_status": "in_progress",
            "llm_calls_used": llm_calls,
            "messages": [AIMessage(content=f"Created refactoring plan with {len(steps)} steps")],
        }

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {
            "overall_status": "failed",
            "error_message": str(e),
            "should_continue": False,
            "llm_calls_used": llm_calls,
            "messages": [AIMessage(content=f"Planning failed: {str(e)}")],
        }


def executor_node(state: AgentState) -> Dict[str, Any]:
    """Executor node that applies refactoring changes via diff/patch application.

    Per file: request a unified diff from the LLM, validate it targets the
    expected path, apply it with `git apply`, then syntax-check the result
    (.py files) and roll back to the pre-patch content if that check fails.
    Nothing is left half-written: either the patch is fully applied and
    valid, or the file is exactly as it was before this attempt.
    """
    from src.tools import read_file, write_file, find_python_files, is_safe_relpath, apply_patch
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logger.info(f"Executing step {state.current_step_index + 1}/{len(state.plan)}")

    if state.current_step_index >= len(state.plan):
        logger.info("All steps completed")
        return {"messages": [AIMessage(content="All execution steps completed")]}

    current_step = state.plan[state.current_step_index]
    logger.info(f"Executing step: {current_step['description']}")

    repo_path = state.repo_path
    new_history = []
    new_processed = list(state.processed_files)
    llm_calls_made = 0

    try:
        if current_step.get('file_path'):
            files_to_process = [current_step['file_path']]
        else:
            files_to_process = find_python_files(repo_path)
            logger.info(f"Found {len(files_to_process)} Python files to process")

        if state.max_files and len(files_to_process) > state.max_files:
            logger.warning(
                f"Capping this step to the first {state.max_files} of {len(files_to_process)} files (--max-files)."
            )
            files_to_process = files_to_process[: state.max_files]

        model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        llm = _get_llm(model_name, temperature=0.3)

        linter_errors = state.linter_errors
        test_failures = state.test_failures
        has_verification_errors = len(linter_errors) > 0 or len(test_failures) > 0

        processed_count = 0
        for file_path in files_to_process:
            if file_path in new_processed and not has_verification_errors:
                continue

            action = current_step.get('action', 'refactor')

            if not is_safe_relpath(repo_path, file_path):
                logger.warning(f"Skipping path outside repo root: {file_path}")
                new_history.append({
                    "step_id": current_step['step_id'],
                    "file_path": file_path,
                    "action": action,
                    "success": False,
                    "error": "Path resolves outside the repository root; refused.",
                })
                continue

            logger.info(f"Processing file: {file_path}")

            result = read_file(file_path)
            if not result.success:
                logger.warning(f"Could not read {file_path}: {result.error}")
                continue

            original_content = result.output
            relative_path = _relpath(file_path, repo_path)

            system_prompt = """You are an expert Python developer. Given a file's content and a \
refactoring task, produce a single unified diff that applies the requested change.

Rules:
- Output ONLY a unified diff - no explanation, no markdown fences, no commentary.
- Use file headers exactly as: "--- a/<path>" and "+++ b/<path>" with the path given to you.
- Include at least 3 lines of context around every change.
- Do not touch lines unrelated to the requested change.
- If no change is actually needed, output an empty response (no diff at all)."""

            task = state.task_description

            user_prompt = f"""File path: {relative_path}

Original content:
{original_content}

Task: {task}
Specific action: {action}
"""

            if has_verification_errors:
                file_errors = [err for err in linter_errors if file_path in err]
                if file_errors:
                    user_prompt += "\nIMPORTANT: The previous version had these linter errors that MUST be fixed:\n"
                    user_prompt += "\n".join(file_errors[:5])
                    user_prompt += "\n\nFix these errors as part of the diff."

            user_prompt += "\nReturn only the unified diff."

            try:
                response = llm.invoke([
                    ("system", system_prompt),
                    ("human", user_prompt)
                ])
                llm_calls_made += 1

                diff_text = _extract_diff(response.content)

                if not diff_text:
                    new_history.append({
                        "step_id": current_step['step_id'],
                        "file_path": file_path,
                        "action": action,
                        "success": True,
                        "note": "No diff produced (no change needed or nothing extractable); file left unchanged.",
                    })
                    continue

                patch_result = apply_patch(repo_path, relative_path, diff_text)
                if not patch_result.success:
                    logger.warning(f"Patch did not apply for {file_path}: {patch_result.error}")
                    new_history.append({
                        "step_id": current_step['step_id'],
                        "file_path": file_path,
                        "action": action,
                        "success": False,
                        "error": f"Patch not applied: {patch_result.error}",
                    })
                    continue

                if file_path.endswith(".py"):
                    post_patch = read_file(file_path)
                    syntax_error = _validate_python_syntax(post_patch.output) if post_patch.success else "could not re-read file"
                    if syntax_error:
                        logger.error(f"Patch applied but produced invalid syntax in {file_path}; reverting: {syntax_error}")
                        write_file(file_path, original_content)
                        new_history.append({
                            "step_id": current_step['step_id'],
                            "file_path": file_path,
                            "action": action,
                            "success": False,
                            "error": f"Patch applied but produced invalid syntax; reverted. {syntax_error}",
                        })
                        continue

                processed_count += 1
                if file_path not in new_processed:
                    new_processed.append(file_path)

                new_history.append({
                    "step_id": current_step['step_id'],
                    "file_path": file_path,
                    "action": action,
                    "success": True,
                    "was_retry": has_verification_errors,
                })

            except Exception as e:
                logger.error(f"Error refactoring {file_path}: {e}")
                new_history.append({
                    "step_id": current_step['step_id'],
                    "file_path": file_path,
                    "action": action,
                    "success": False,
                    "error": str(e),
                })

        logger.info(f"Processed {processed_count} files in this step")

        next_index = state.current_step_index + 1

        return {
            "current_step_index": next_index,
            "processed_files": new_processed,
            "execution_history": new_history,
            "llm_calls_used": state.llm_calls_used + llm_calls_made,
            "messages": [AIMessage(content=f"Executed step {current_step['step_id']}: {current_step['description']}")],
        }

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return {
            "overall_status": "failed",
            "error_message": str(e),
            "should_continue": False,
            "execution_history": new_history,
            "llm_calls_used": state.llm_calls_used + llm_calls_made,
            "messages": [AIMessage(content=f"Execution failed: {str(e)}")],
        }


def verifier_node(state: AgentState) -> Dict[str, Any]:
    """Verifier node that runs linters and tests, sandboxed by default.

    Owns the retry-loop bound: each time issues are found, retry_count is
    incremented; once it exceeds max_retries, the workflow ends "failed"
    instead of looping indefinitely.
    """
    from src.tools import run_linter, run_tests, find_python_files, is_docker_available

    logger.info("Running verification checks")

    repo_path = state.repo_path
    sandbox_enabled = state.sandbox_enabled
    max_retries = state.max_retries
    retry_count = state.retry_count

    if sandbox_enabled and not is_docker_available():
        message = (
            "Sandboxed verification is enabled but Docker is not available on this "
            "host. Refusing to run lint/test commands from an untrusted repository "
            "unsandboxed. Install Docker, or re-run with --no-sandbox to explicitly "
            "accept the risk."
        )
        logger.error(message)
        return {
            "overall_status": "failed",
            "error_message": message,
            "should_continue": False,
            "messages": [AIMessage(content=message)],
        }

    try:
        python_files = find_python_files(repo_path)

        linter_errors = []
        for file_path in python_files:
            result = run_linter(file_path, "ruff", repo_root=repo_path, use_sandbox=sandbox_enabled)
            if not result.success:
                linter_errors.append(f"{file_path}: {result.output or result.error}")

        for file_path in python_files:
            result = run_linter(file_path, "black", repo_root=repo_path, use_sandbox=sandbox_enabled)
            if not result.success:
                linter_errors.append(f"{file_path}: {result.output or result.error}")

        test_failures = []
        test_result = run_tests(".", cwd=repo_path, use_sandbox=sandbox_enabled)
        if not test_result.success:
            test_failures.append((test_result.output or test_result.error or "")[:1000])

        logger.info(f"Verification complete. Linter errors: {len(linter_errors)}, Test failures: {len(test_failures)}")

        has_issues = len(linter_errors) > 0 or len(test_failures) > 0

        if has_issues:
            new_retry_count = retry_count + 1

            verification_entry = [{
                "type": "verification",
                "linter_errors": linter_errors[:5],
                "test_failures": test_failures[:3],
                "requires_fix": True,
                "retry_count": new_retry_count,
                "success": False,
            }]

            if new_retry_count > max_retries:
                message = (
                    f"Verification still failing after {max_retries} retries "
                    f"({len(linter_errors)} linter issues, {len(test_failures)} test failures remaining). "
                    "Stopping instead of retrying indefinitely."
                )
                logger.error(message)
                return {
                    "linter_errors": linter_errors,
                    "test_failures": test_failures,
                    "retry_count": new_retry_count,
                    "overall_status": "failed",
                    "error_message": message,
                    "should_continue": False,
                    "execution_history": verification_entry,
                    "messages": [AIMessage(content=message)],
                }

            return {
                "linter_errors": linter_errors,
                "test_failures": test_failures,
                "retry_count": new_retry_count,
                "overall_status": "verification_failed",
                "should_continue": True,
                "execution_history": verification_entry,
                "messages": [AIMessage(
                    content=f"Verification found {len(linter_errors)} linter issues and {len(test_failures)} "
                            f"test failures (retry {new_retry_count}/{max_retries}). Errors: {linter_errors[:3]}"
                )],
            }
        else:
            return {
                "linter_errors": [],
                "test_failures": [],
                "overall_status": "awaiting_approval",
                "should_continue": True,
                "messages": [AIMessage(content="Verification passed. Changes are ready for review.")],
            }

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {
            "overall_status": "failed",
            "error_message": str(e),
            "should_continue": False,
            "messages": [AIMessage(content=f"Verification failed: {str(e)}")],
        }


def github_integration_node(state, github_token: Optional[str] = None) -> Dict[str, Any]:
    """Create a branch, commit, push, and open a PR.

    Not wired into the automatic graph flow - called directly by the CLI,
    only after the operator has reviewed a diff and explicitly approved it
    (see main.py).

    Args:
        state: Current agent state. Accepted as either a plain dict or an
            `AgentState` instance and normalized to a dict internally,
            since LangGraph's exact return representation for
            Pydantic-schema graphs can vary by version (see
            src/state/models.py's version note) and this function is also
            called directly, outside the graph.
        github_token: GitHub token, passed explicitly rather than read from
            the environment here, so callers control exactly when/whether
            it's used and it's easier to keep out of logs.
    """
    from src.tools import create_git_branch, commit_changes, push_branch, create_pull_request, run_command, redact_secrets, is_valid_branch_name
    from datetime import datetime

    state = state if isinstance(state, dict) else state.model_dump()

    logger.info("Starting GitHub integration")

    repo_path = state['repo_path']
    branch_name = state['branch_name']
    token = github_token

    def _err(message: str) -> Dict[str, Any]:
        safe_message = redact_secrets(message, [token] if token else [])
        return {
            **state,
            "overall_status": "failed",
            "error_message": safe_message,
            "messages": state["messages"] + [AIMessage(content=safe_message)],
        }

    if not is_valid_branch_name(branch_name):
        return _err(f"Refusing unsafe branch name: {branch_name!r}")

    try:
        branch_result = create_git_branch(branch_name, repo_path)
        if not branch_result.success:
            checkout_result = run_command(["git", "checkout", "-b", branch_name], cwd=repo_path)
            if not checkout_result.success:
                return _err(f"Failed to create branch: {branch_result.error}")

        logger.info(f"Created branch: {branch_name}")

        commit_msg = f"Refactor: {state['task_description']}\n\nAutomated refactoring by Codebase Refactor Agent"
        commit_result = commit_changes(commit_msg, repo_path)
        if not commit_result.success:
            return _err(f"Failed to commit: {commit_result.error}")

        logger.info("Committed changes")

        push_result = push_branch(branch_name, repo_path, token=token)
        if not push_result.success:
            return _err(f"Failed to push: {push_result.error}")

        logger.info(f"Pushed branch: {branch_name}")

        pr_title = f"Refactor: {state['task_description']}"
        processed_files = state['processed_files']
        pr_description = f"""## Automated Refactoring

This PR was created automatically by the Codebase Refactor Agent, and was
reviewed and approved by a human operator before being pushed. Changes were
applied as diffs/patches, not whole-file rewrites.

### Task
{state['task_description']}

### Changes
- Processed {len(processed_files)} files
- {state.get('llm_calls_used', 0)} LLM calls used this run

### Files Modified
{chr(10).join(f'- `{f}`' for f in processed_files[:20])}
{'... and more' if len(processed_files) > 20 else ''}

### Verification
- Linting: {'✅ Passed' if not state['linter_errors'] else '❌ Issues found'}
- Tests: {'✅ Passed' if not state['test_failures'] else '❌ Failures detected'}

---
*Generated by Codebase Refactor Agent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        if token:
            pr_success, pr_msg, pr_url = create_pull_request(
                repo_url=state['repo_url'],
                branch_name=branch_name,
                title=pr_title,
                description=pr_description,
                token=token
            )

            if pr_success:
                logger.info(f"Created PR: {pr_url}")
                return {
                    **state,
                    "pr_title": pr_title,
                    "pr_description": pr_description,
                    "pr_url": pr_url,
                    "overall_status": "completed",
                    "messages": state["messages"] + [AIMessage(content=f"Pull request created: {pr_url}")],
                }
            else:
                safe_msg = redact_secrets(pr_msg, [token])
                return {
                    **state,
                    "overall_status": "completed_no_pr",
                    "error_message": safe_msg,
                    "messages": state["messages"] + [AIMessage(content=f"Pushed branch, but PR creation failed: {safe_msg}")],
                }
        else:
            return {
                **state,
                "pr_title": pr_title,
                "pr_description": pr_description,
                "overall_status": "completed_no_pr",
                "error_message": "No GitHub token provided, skipping PR creation",
                "messages": state["messages"] + [AIMessage(content="Changes committed and pushed. No GitHub token provided, skipping PR creation.")],
            }

    except Exception as e:
        return _err(f"GitHub integration failed: {str(e)}")


def build_refactor_graph(checkpointer=None):
    """Build the LangGraph workflow for codebase refactoring.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. a SQLite
            checkpointer) enabling a run to be resumed by thread_id after a
            crash. See main.py's `--resume` flag. If omitted, the graph
            still works exactly as before - checkpointing is additive.

    The graph covers planning, execution, and (bounded) verification/retry.
    It intentionally ends at `awaiting_approval` rather than continuing on
    into commit/push/PR - that step requires human approval and is invoked
    directly by the caller (see main.py) once the operator has reviewed the
    diff.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "verifier",
        lambda state: "retry" if (state.get('overall_status') if isinstance(state, dict) else state.overall_status) == 'verification_failed' else "end",
        {
            "retry": "executor",
            "end": END,
        }
    )
    workflow.add_edge("executor", "verifier")

    return workflow.compile(checkpointer=checkpointer)
