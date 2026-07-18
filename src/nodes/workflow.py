"""LangGraph nodes for the Codebase Refactor Agent.

Security-relevant changes from the original implementation
------------------------------------------------------------
1. The graph no longer auto-commits, pushes, and opens a PR the moment
   verification passes. `verifier_node`'s success path ends the graph in an
   `"awaiting_approval"` state. `github_integration_node` still exists and
   does the same job as before, but it is now called *directly by the CLI*,
   only after the operator has reviewed a diff and explicitly approved it
   (or passed `--yes`). See main.py.
2. `executor_node` validates that any file it's about to touch stays inside
   the cloned repository (no path traversal), and syntax-checks Python
   output with `ast.parse` before writing it - if the model's response
   isn't valid Python, the file is left untouched and the failure is
   recorded, instead of silently corrupting the file on disk.
3. `verifier_node` runs linting/formatting/tests through the sandbox
   (src/tools/sandbox.py) by default, since those commands execute code
   that lives inside the untrusted cloned repository. It also removes the
   original "first 5 / first 3 files, for demo" shortcut, and bounds the
   executor<->verifier retry loop via `max_retries` (which existed as an
   unused field in the Pydantic model before, but was never wired up).
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class AgentState(TypedDict):
    """State for the LangGraph workflow."""

    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    repo_url: str
    repo_path: str
    task_description: str
    target_files: list[str] | None
    branch_name: str

    # Planning
    plan: list[dict]
    current_step_index: int

    # Execution tracking
    processed_files: list[str]
    execution_history: list[dict]

    # Verification results
    linter_errors: list[str]
    test_failures: list[str]
    retry_count: int
    max_retries: int
    sandbox_enabled: bool

    # GitHub integration (populated only after human approval, see main.py)
    pr_title: str | None
    pr_description: str | None
    pr_url: str | None

    # Overall status
    overall_status: str
    error_message: str | None
    should_continue: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(llm_text: str) -> str | None:
    """Pull code out of an LLM response, handling the common formatting cases.

    The original implementation only stripped fences when the response
    *started* with ``` - if the model prefaced its answer with any
    commentary ("Here's the refactored code:"), that commentary got written
    into the file verbatim. This checks for a fenced block anywhere in the
    response first, and falls back to the raw (stripped) text only if no
    fence is found at all.
    """
    if not llm_text:
        return None

    text = llm_text.strip()

    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    if text.startswith("```") or text.endswith("```"):
        # Malformed / partial fence - don't guess, treat as unusable.
        return None

    return text if text else None


def _validate_python_syntax(content: str) -> str | None:
    """Return an error message if `content` isn't valid Python, else None."""
    try:
        ast.parse(content)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"


def _get_llm(model_name: str, temperature: float):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model_name, temperature=temperature)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> AgentState:
    """Planner node that breaks down the refactoring task into steps."""
    import os
    import json
    from dotenv import load_dotenv

    load_dotenv()

    logger.info(f"Planning refactoring task: {state['task_description']}")

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

    user_prompt = f"""Task: {state['task_description']}
Repository: {state['repo_url']}
Target files: {state['target_files'] or 'All Python files'}

Create a detailed refactoring plan as a JSON array."""

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
                {"step_id": 1, "description": f"Analyze codebase for {state['task_description']}", "file_path": None, "action": "analyze"},
                {"step_id": 2, "description": f"Apply refactoring: {state['task_description']}", "file_path": None, "action": "refactor"},
                {"step_id": 3, "description": "Run linters and formatters", "file_path": None, "action": "verify"},
            ]

        logger.info(f"Created plan with {len(steps)} steps")

        return {
            **state,
            "plan": steps,
            "current_step_index": 0,
            "overall_status": "in_progress",
            "messages": state["messages"] + [AIMessage(content=f"Created refactoring plan with {len(steps)} steps")]
        }

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {
            **state,
            "overall_status": "failed",
            "error_message": str(e),
            "should_continue": False,
            "messages": state["messages"] + [AIMessage(content=f"Planning failed: {str(e)}")]
        }


def executor_node(state: AgentState) -> AgentState:
    """Executor node that applies refactoring changes.

    Every write is guarded by two checks that didn't exist before:
      * the target path must resolve inside the cloned repo (no traversal)
      * for .py files, the LLM's output must parse as valid Python before
        it's written - if it doesn't, the original file is left alone.
    """
    from src.tools import read_file, write_file, find_python_files, is_safe_relpath
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logger.info(f"Executing step {state['current_step_index'] + 1}/{len(state['plan'])}")

    if state['current_step_index'] >= len(state['plan']):
        logger.info("All steps completed")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="All execution steps completed")]
        }

    current_step = state['plan'][state['current_step_index']]
    logger.info(f"Executing step: {current_step['description']}")

    repo_path = state['repo_path']

    try:
        if current_step.get('file_path'):
            files_to_process = [current_step['file_path']]
        else:
            files_to_process = find_python_files(repo_path)
            logger.info(f"Found {len(files_to_process)} Python files to process")

        model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        llm = _get_llm(model_name, temperature=0.3)

        linter_errors = state.get('linter_errors', [])
        test_failures = state.get('test_failures', [])
        has_verification_errors = len(linter_errors) > 0 or len(test_failures) > 0

        processed_count = 0
        for file_path in files_to_process:
            if file_path in state['processed_files'] and not has_verification_errors:
                continue

            if not is_safe_relpath(repo_path, file_path):
                logger.warning(f"Skipping path outside repo root: {file_path}")
                state['execution_history'].append({
                    "step_id": current_step['step_id'],
                    "file_path": file_path,
                    "action": current_step.get('action', 'refactor'),
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

            system_prompt = """You are an expert Python developer. Your task is to refactor code according to the given instructions.
Return ONLY the refactored code, no explanations or markdown."""

            action = current_step.get('action', 'refactor')
            task = state['task_description']

            user_prompt = f"""Original code:
{original_content}

Task: {task}
Specific action: {action}
"""

            if has_verification_errors:
                file_errors = [err for err in linter_errors if file_path in err]
                if file_errors:
                    user_prompt += "\nIMPORTANT: The previous version had these linter errors that MUST be fixed:\n"
                    user_prompt += "\n".join(file_errors[:5])
                    user_prompt += "\n\nFix these errors while applying the refactoring."

            user_prompt += "\nRefactor the code accordingly. Return only the complete refactored code."

            try:
                response = llm.invoke([
                    ("system", system_prompt),
                    ("human", user_prompt)
                ])

                refactored_content = _extract_code(response.content)

                if not refactored_content:
                    logger.warning(f"Model returned no usable code for {file_path}; leaving file unchanged")
                    state['execution_history'].append({
                        "step_id": current_step['step_id'],
                        "file_path": file_path,
                        "action": action,
                        "success": False,
                        "error": "Model response contained no extractable code; file left unchanged.",
                    })
                    continue

                if file_path.endswith(".py"):
                    syntax_error = _validate_python_syntax(refactored_content)
                    if syntax_error:
                        logger.error(f"Refusing to write invalid Python to {file_path}: {syntax_error}")
                        state['execution_history'].append({
                            "step_id": current_step['step_id'],
                            "file_path": file_path,
                            "action": action,
                            "success": False,
                            "error": f"Generated content failed syntax check, file left unchanged: {syntax_error}",
                        })
                        continue

                write_result = write_file(file_path, refactored_content)
                if write_result.success:
                    processed_count += 1
                    if file_path not in state['processed_files']:
                        state['processed_files'].append(file_path)

                    state['execution_history'].append({
                        "step_id": current_step['step_id'],
                        "file_path": file_path,
                        "action": action,
                        "success": True,
                        "was_retry": has_verification_errors
                    })
                else:
                    logger.error(f"Failed to write {file_path}: {write_result.error}")

            except Exception as e:
                logger.error(f"Error refactoring {file_path}: {e}")
                state['execution_history'].append({
                    "step_id": current_step['step_id'],
                    "file_path": file_path,
                    "action": action,
                    "success": False,
                    "error": str(e)
                })

        logger.info(f"Processed {processed_count} files in this step")

        next_index = state['current_step_index'] + 1

        return {
            **state,
            "current_step_index": next_index,
            "messages": state["messages"] + [AIMessage(content=f"Executed step {current_step['step_id']}: {current_step['description']}")]
        }

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return {
            **state,
            "overall_status": "failed",
            "error_message": str(e),
            "should_continue": False,
            "messages": state["messages"] + [AIMessage(content=f"Execution failed: {str(e)}")]
        }


def verifier_node(state: AgentState) -> AgentState:
    """Verifier node that runs linters and tests, sandboxed by default.

    Also owns the retry-loop bound: each time issues are found, retry_count
    is incremented; once it reaches max_retries, the workflow ends with a
    "failed" status instead of looping indefinitely.
    """
    from src.tools import run_linter, run_tests, find_python_files, is_docker_available

    logger.info("Running verification checks")

    repo_path = state['repo_path']
    sandbox_enabled = state.get('sandbox_enabled', True)
    max_retries = state.get('max_retries', 3)
    retry_count = state.get('retry_count', 0)

    if sandbox_enabled and not is_docker_available():
        message = (
            "Sandboxed verification is enabled but Docker is not available on this "
            "host. Refusing to run lint/test commands from an untrusted repository "
            "unsandboxed. Install Docker, or re-run with --no-sandbox to explicitly "
            "accept the risk."
        )
        logger.error(message)
        return {
            **state,
            "overall_status": "failed",
            "error_message": message,
            "should_continue": False,
            "messages": state["messages"] + [AIMessage(content=message)]
        }

    try:
        python_files = find_python_files(repo_path)

        linter_errors: list[str] = []

        for file_path in python_files:
            result = run_linter(file_path, "ruff", repo_root=repo_path, use_sandbox=sandbox_enabled)
            if not result.success:
                linter_errors.append(f"{file_path}: {result.output or result.error}")

        for file_path in python_files:
            result = run_linter(file_path, "black", repo_root=repo_path, use_sandbox=sandbox_enabled)
            if not result.success:
                linter_errors.append(f"{file_path}: {result.output or result.error}")

        test_failures: list[str] = []
        test_result = run_tests(".", cwd=repo_path, use_sandbox=sandbox_enabled)
        if not test_result.success:
            test_failures.append((test_result.output or test_result.error or "")[:1000])

        logger.info(f"Verification complete. Linter errors: {len(linter_errors)}, Test failures: {len(test_failures)}")

        has_issues = len(linter_errors) > 0 or len(test_failures) > 0

        if has_issues:
            new_retry_count = retry_count + 1

            state['execution_history'].append({
                "type": "verification",
                "linter_errors": linter_errors[:5],
                "test_failures": test_failures[:3],
                "requires_fix": True,
                "retry_count": new_retry_count,
            })

            if new_retry_count > max_retries:
                message = (
                    f"Verification still failing after {max_retries} retries "
                    f"({len(linter_errors)} linter issues, {len(test_failures)} test failures remaining). "
                    "Stopping instead of retrying indefinitely."
                )
                logger.error(message)
                return {
                    **state,
                    "linter_errors": linter_errors,
                    "test_failures": test_failures,
                    "retry_count": new_retry_count,
                    "overall_status": "failed",
                    "error_message": message,
                    "should_continue": False,
                    "messages": state["messages"] + [AIMessage(content=message)]
                }

            return {
                **state,
                "linter_errors": linter_errors,
                "test_failures": test_failures,
                "retry_count": new_retry_count,
                "overall_status": "verification_failed",
                "should_continue": True,
                "messages": state["messages"] + [AIMessage(
                    content=f"Verification found {len(linter_errors)} linter issues and {len(test_failures)} "
                            f"test failures (retry {new_retry_count}/{max_retries}). Errors: {linter_errors[:3]}"
                )]
            }
        else:
            return {
                **state,
                "linter_errors": [],
                "test_failures": [],
                "overall_status": "awaiting_approval",
                "should_continue": True,
                "messages": state["messages"] + [AIMessage(content="Verification passed. Changes are ready for review.")]
            }

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {
            **state,
            "overall_status": "failed",
            "error_message": str(e),
            "should_continue": False,
            "messages": state["messages"] + [AIMessage(content=f"Verification failed: {str(e)}")]
        }


def github_integration_node(state: AgentState, github_token: str | None = None) -> AgentState:
    """Create a branch, commit, push, and open a PR.

    This is *not* wired into the automatic graph flow anymore - it's called
    directly by the CLI, and only after the operator has reviewed a diff of
    the working tree and explicitly approved it (see main.py). This is the
    fix for "blind auto-commit/push/PR": the agent no longer ships changes
    nobody has looked at.

    Args:
        state: Current agent state (must have passed verification).
        github_token: GitHub token, passed explicitly rather than read from
            the environment here, so callers control exactly when/whether
            it's used and it's easier to keep out of logs.
    """
    from src.tools import create_git_branch, commit_changes, push_branch, create_pull_request, run_command, redact_secrets, is_valid_branch_name
    from datetime import datetime

    logger.info("Starting GitHub integration")

    repo_path = state['repo_path']
    branch_name = state['branch_name']
    token = github_token

    def _err(message: str) -> AgentState:
        safe_message = redact_secrets(message, [token] if token else [])
        return {
            **state,
            "overall_status": "failed",
            "error_message": safe_message,
            "messages": state["messages"] + [AIMessage(content=safe_message)]
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
reviewed and approved by a human operator before being pushed.

### Task
{state['task_description']}

### Changes
- Processed {len(processed_files)} files
- Applied automated refactoring based on best practices

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
                    "messages": state["messages"] + [AIMessage(content=f"Pull request created: {pr_url}")]
                }
            else:
                safe_msg = redact_secrets(pr_msg, [token])
                return {
                    **state,
                    # Distinct from "completed": the push succeeded but the PR
                    # did not, so this shouldn't read as fully successful.
                    "overall_status": "completed_no_pr",
                    "error_message": safe_msg,
                    "messages": state["messages"] + [AIMessage(content=f"Pushed branch, but PR creation failed: {safe_msg}")]
                }
        else:
            return {
                **state,
                "pr_title": pr_title,
                "pr_description": pr_description,
                "overall_status": "completed_no_pr",
                "error_message": "No GitHub token provided, skipping PR creation",
                "messages": state["messages"] + [AIMessage(content="Changes committed and pushed. No GitHub token provided, skipping PR creation.")]
            }

    except Exception as e:
        return _err(f"GitHub integration failed: {str(e)}")


def should_continue(state: AgentState) -> str:
    """Conditional edge to determine if workflow should continue."""
    if state.get('should_continue') is False:
        return "end"
    if state.get('overall_status') == 'failed':
        return "end"
    if state.get('overall_status') in ('completed', 'completed_no_pr'):
        return "end"
    return "continue"


def build_refactor_graph():
    """Build the LangGraph workflow for codebase refactoring.

    The graph covers planning, execution, and (bounded) verification/retry.
    It intentionally ends at `awaiting_approval` rather than continuing on
    into commit/push/PR - that step requires human approval and is invoked
    directly by the caller (see main.py) once the operator has reviewed the
    diff. This keeps "propose changes" and "ship changes" as separate,
    independently-auditable phases.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "verifier",
        lambda state: "retry" if state.get('overall_status') == 'verification_failed' else "end",
        {
            "retry": "executor",
            "end": END,
        }
    )
    workflow.add_edge("executor", "verifier")

    return workflow.compile()
