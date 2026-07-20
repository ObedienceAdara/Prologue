"""State models for the Codebase Refactor Agent.

This used to be two divergent definitions: a Pydantic `AgentState` here
(unused by the graph) and a separate `TypedDict AgentState` in
src/nodes/workflow.py (what the graph actually ran on). They're now
reconciled into this single model, which is what LangGraph is compiled
against (`StateGraph(AgentState)` in workflow.py).

Version note
------------
LangGraph's support for Pydantic-model-as-state is a documented feature,
but its exact behavior (e.g. whether `.invoke()` returns a plain dict or a
model instance) has shifted across versions and wasn't verifiable against a
live install while writing this. Callers that consume the final state
(see main.py) normalize it to a plain dict at the boundary
(`state if isinstance(state, dict) else state.model_dump()`) specifically
to stay correct either way. If you hit an incompatibility, that
normalization point is the first place to look.

Reducers
--------
`messages` and `execution_history` are marked `Annotated[..., operator.add]`,
meaning LangGraph concatenates whatever a node returns for that field onto
the existing value, rather than replacing it - each node only needs to
return the *new* messages/history entries it produced, not the full
accumulated list. Every other field uses default (replace) semantics: a
node returning a value for that field replaces it outright.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import BaseMessage


class TaskStatus(str, Enum):
    """Status values used for individual refactoring steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class RefactoringStep(BaseModel):
    """A single step in the refactoring plan."""
    step_id: int
    description: str
    file_path: Optional[str] = None
    action: str = "refactor"
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0


class AgentState(BaseModel):
    """Canonical state for the LangGraph refactoring workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Conversation / audit trail. Reducer: each node returns only the new
    # message(s) it produced; LangGraph appends them to the running list.
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)

    # Task setup
    repo_url: str
    repo_path: str = ""
    task_description: str
    target_files: Optional[List[str]] = None
    branch_name: str = ""

    # Planning
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    current_step_index: int = 0

    # Execution tracking. `processed_files` is a plain (replace-semantics)
    # field - the executor computes and returns the full deduplicated list
    # each time, since "already processed" membership matters.
    # `execution_history` uses the operator.add reducer - nodes append-only.
    processed_files: List[str] = Field(default_factory=list)
    execution_history: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)

    # Verification
    linter_errors: List[str] = Field(default_factory=list)
    test_failures: List[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    sandbox_enabled: bool = True

    # Cost / scope controls
    max_files: Optional[int] = None
    llm_calls_used: int = 0

    # GitHub integration (populated only after human approval - see main.py)
    pr_title: Optional[str] = None
    pr_description: Optional[str] = None
    pr_url: Optional[str] = None

    # Overall status
    overall_status: str = "pending"
    error_message: Optional[str] = None
    should_continue: bool = True
