"""LangGraph nodes for the Codebase Refactor Agent."""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


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
    
    # GitHub integration
    pr_title: str | None
    pr_description: str | None
    pr_url: str | None
    
    # Overall status
    overall_status: str
    error_message: str | None
    should_continue: bool


def planner_node(state: AgentState) -> AgentState:
    """Planner node that breaks down the refactoring task into steps.
    
    This node analyzes the task description and creates a multi-step plan
    for refactoring the codebase.
    """
    from src.state.models import AgentState as StateModel, RefactoringStep, TaskStatus
    from langchain_groq import ChatGroq
    import os
    import json
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logger.info(f"Planning refactoring task: {state['task_description']}")
    
    # Initialize LLM with Groq
    model_name = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    llm = ChatGroq(model=model_name, temperature=0)
    
    # Create planning prompt with explicit JSON output instruction
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
        
        # Parse the response - extract JSON from the content
        plan_text = response.content.strip()
        
        # Remove markdown code blocks if present
        if plan_text.startswith("```"):
            lines = plan_text.split("\n")
            # Find the first line that doesn't start with ```
            start_idx = 0
            for i, line in enumerate(lines):
                if not line.startswith("```"):
                    start_idx = i
                    break
            # Find the last line that doesn't start with ```
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if not lines[i].startswith("```"):
                    end_idx = i + 1
                    break
            plan_text = "\n".join(lines[start_idx:end_idx])
        
        # Try to parse the JSON response
        steps = []
        try:
            parsed_steps = json.loads(plan_text)
            if isinstance(parsed_steps, list):
                # Validate and normalize the steps
                for i, step in enumerate(parsed_steps):
                    if isinstance(step, dict):
                        validated_step = {
                            "step_id": step.get("step_id", i + 1),
                            "description": step.get("description", f"Step {i + 1}"),
                            "file_path": step.get("file_path"),
                            "action": step.get("action", "refactor")
                        }
                        steps.append(validated_step)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON, using fallback plan")
        
        # If parsing failed or returned empty, create a sensible default plan
        if not steps:
            steps = [
                {
                    "step_id": 1,
                    "description": f"Analyze codebase for {state['task_description']}",
                    "file_path": None,
                    "action": "analyze"
                },
                {
                    "step_id": 2,
                    "description": f"Apply refactoring: {state['task_description']}",
                    "file_path": None,
                    "action": "refactor"
                },
                {
                    "step_id": 3,
                    "description": "Run linters and formatters",
                    "file_path": None,
                    "action": "verify"
                }
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
    
    This node executes the current step in the plan by calling appropriate tools
    to modify the codebase. It also handles self-correction when verification
    errors are passed back from the verifier node.
    """
    from src.tools import read_file, write_file, run_linter, run_formatter, find_python_files
    from langchain_groq import ChatGroq
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
    
    try:
        # Determine which files to process
        if current_step.get('file_path'):
            files_to_process = [current_step['file_path']]
        else:
            # Find all Python files in the repo
            files_to_process = find_python_files(state['repo_path'])
            logger.info(f"Found {len(files_to_process)} Python files to process")
        
        # Initialize LLM for code transformation with Groq
        model_name = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        llm = ChatGroq(model=model_name, temperature=0.3)
        
        # Check if we're in a retry loop due to verification failures
        linter_errors = state.get('linter_errors', [])
        test_failures = state.get('test_failures', [])
        has_verification_errors = len(linter_errors) > 0 or len(test_failures) > 0
        
        processed_count = 0
        for file_path in files_to_process:
            if file_path in state['processed_files'] and not has_verification_errors:
                continue
                
            logger.info(f"Processing file: {file_path}")
            
            # Read the file
            result = read_file(file_path)
            if not result.success:
                logger.warning(f"Could not read {file_path}: {result.error}")
                continue
            
            original_content = result.output
            
            # Create refactoring prompt - include error context if available
            system_prompt = """You are an expert Python developer. Your task is to refactor code according to the given instructions.
Return ONLY the refactored code, no explanations or markdown."""
            
            action = current_step.get('action', 'refactor')
            task = state['task_description']
            
            # Build user prompt with optional error context for self-correction
            user_prompt = f"""Original code:
{original_content}

Task: {task}
Specific action: {action}
"""
            
            # Add error context if we're in a retry loop
            if has_verification_errors:
                # Find errors specific to this file
                file_errors = [err for err in linter_errors if file_path in err]
                if file_errors:
                    user_prompt += f"\nIMPORTANT: The previous version had these linter errors that MUST be fixed:\n"
                    user_prompt += "\n".join(file_errors[:5])
                    user_prompt += "\n\nFix these errors while applying the refactoring."
            
            user_prompt += "\nRefactor the code accordingly. Return only the complete refactored code."
            
            try:
                response = llm.invoke([
                    ("system", system_prompt),
                    ("human", user_prompt)
                ])
                
                refactored_content = response.content.strip()
                
                # Remove markdown code blocks if present
                if refactored_content.startswith("```"):
                    lines = refactored_content.split("\n")
                    if lines[0].startswith("```python") or lines[0] == "```":
                        refactored_content = "\n".join(lines[1:-1])
                
                # Write the refactored code
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
        
        # Move to next step
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
    """Verifier node that runs linters and tests.
    
    This node validates the refactored code by running linters and tests.
    If issues are found, it can loop back to the executor for fixes.
    """
    from src.tools import run_linter, run_formatter, run_tests, find_python_files
    from langchain_core.messages import AIMessage
    import os
    
    logger.info("Running verification checks")
    
    try:
        # Find all Python files
        python_files = find_python_files(state['repo_path'])
        
        linter_errors = []
        
        # Run ruff on each file
        for file_path in python_files[:5]:  # Limit to first 5 files for demo
            result = run_linter(file_path, "ruff")
            if not result.success:
                linter_errors.append(f"{file_path}: {result.output}")
        
        # Run black check
        for file_path in python_files[:3]:  # Limit for demo
            result = run_linter(file_path, "black")
            if not result.success:
                linter_errors.append(f"{file_path}: {result.output}")
        
        # Run tests if they exist
        test_failures = []
        test_result = run_tests(".", state['repo_path'])
        if not test_result.success:
            test_failures.append(test_result.output[:1000])  # Truncate long outputs
        
        logger.info(f"Verification complete. Linter errors: {len(linter_errors)}, Test failures: {len(test_failures)}")
        
        # Determine if we need to loop back
        has_issues = len(linter_errors) > 0 or len(test_failures) > 0
        
        if has_issues:
            # Add verification feedback to execution history with error details for self-correction
            state['execution_history'].append({
                "type": "verification",
                "linter_errors": linter_errors[:5],  # Limit errors stored
                "test_failures": test_failures[:3],
                "requires_fix": True
            })
            
            return {
                **state,
                "linter_errors": linter_errors,
                "test_failures": test_failures,
                "overall_status": "verification_failed",
                "should_continue": True,  # Explicitly continue to allow retry
                "messages": state["messages"] + [AIMessage(content=f"Verification found {len(linter_errors)} linter issues and {len(test_failures)} test failures. Errors: {linter_errors[:3]}")]
            }
        else:
            return {
                **state,
                "linter_errors": [],
                "test_failures": [],
                "overall_status": "verified",
                "should_continue": True,
                "messages": state["messages"] + [AIMessage(content="Verification passed successfully")]
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


def github_integration_node(state: AgentState) -> AgentState:
    """Node that creates branch, commits changes, and opens PR.
    
    This node handles all GitHub integration tasks after successful refactoring.
    """
    from src.tools import create_git_branch, commit_changes, push_branch, create_pull_request
    import os
    from dotenv import load_dotenv
    from datetime import datetime
    
    load_dotenv()
    
    logger.info("Starting GitHub integration")
    
    try:
        repo_path = state['repo_path']
        branch_name = state['branch_name']
        token = os.getenv("GITHUB_TOKEN")
        
        # Create and checkout branch
        branch_result = create_git_branch(branch_name, repo_path)
        if not branch_result.success:
            # Branch might already exist, try to checkout
            from src.tools import run_command
            checkout_result = run_command(f"git checkout -b {branch_name}", cwd=repo_path)
            if not checkout_result.success:
                return {
                    **state,
                    "overall_status": "failed",
                    "error_message": f"Failed to create branch: {branch_result.error}",
                    "messages": state["messages"] + [AIMessage(content=f"Branch creation failed: {branch_result.error}")]
                }
        
        logger.info(f"Created branch: {branch_name}")
        
        # Commit changes
        commit_msg = f"Refactor: {state['task_description']}\n\nAutomated refactoring by Codebase Refactor Agent"
        commit_result = commit_changes(commit_msg, repo_path)
        if not commit_result.success:
            return {
                **state,
                "overall_status": "failed",
                "error_message": f"Failed to commit: {commit_result.error}",
                "messages": state["messages"] + [AIMessage(content=f"Commit failed: {commit_result.error}")]
            }
        
        logger.info("Committed changes")
        
        # Push branch
        push_result = push_branch(branch_name, repo_path)
        if not push_result.success:
            return {
                **state,
                "overall_status": "failed",
                "error_message": f"Failed to push: {push_result.error}",
                "messages": state["messages"] + [AIMessage(content=f"Push failed: {push_result.error}")]
            }
        
        logger.info(f"Pushed branch: {branch_name}")
        
        # Create pull request
        pr_title = f"Refactor: {state['task_description']}"
        
        # Generate PR description
        processed_files = state['processed_files']
        pr_description = f"""## Automated Refactoring

This PR was created automatically by the Codebase Refactor Agent.

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
                return {
                    **state,
                    "overall_status": "completed",  # Still mark as completed even if PR creation fails
                    "error_message": pr_msg,
                    "messages": state["messages"] + [AIMessage(content=f"PR creation failed: {pr_msg}")]
                }
        else:
            return {
                **state,
                "pr_title": pr_title,
                "pr_description": pr_description,
                "overall_status": "completed",
                "error_message": "No GitHub token provided, skipping PR creation",
                "messages": state["messages"] + [AIMessage(content="Changes committed and pushed. No GitHub token provided, skipping PR creation.")]
            }
        
    except Exception as e:
        logger.error(f"GitHub integration failed: {e}")
        return {
            **state,
            "overall_status": "failed",
            "error_message": str(e),
            "messages": state["messages"] + [AIMessage(content=f"GitHub integration failed: {str(e)}")]
        }


def should_continue(state: AgentState) -> str:
    """Conditional edge to determine if workflow should continue."""
    if state.get('should_continue') is False:
        return "end"
    if state.get('overall_status') == 'failed':
        return "end"
    if state.get('overall_status') == 'completed':
        return "end"
    return "continue"


def build_refactor_graph():
    """Build the LangGraph workflow for codebase refactoring."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("github_integration", github_integration_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "verifier",
        lambda state: "retry" if state.get('overall_status') == 'verification_failed' else "proceed",
        {
            "retry": "executor",
            "proceed": "github_integration"
        }
    )
    workflow.add_edge("executor", "verifier")
    workflow.add_edge("github_integration", END)
    
    return workflow.compile()
