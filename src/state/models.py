"""State models for the Codebase Refactor Agent."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a refactoring task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"


class RefactoringStep(BaseModel):
    """A single step in the refactoring plan."""
    step_id: int = Field(..., description="Unique identifier for the step")
    description: str = Field(..., description="Description of what this step does")
    file_path: Optional[str] = Field(None, description="Path to the file being modified")
    action: str = Field(..., description="Action to perform (e.g., 'add_type_hints', 'extract_method')")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of this step")
    error_message: Optional[str] = Field(None, description="Error message if the step failed")
    retry_count: int = Field(default=0, description="Number of times this step has been retried")


class AgentState(BaseModel):
    """Main state model for the refactoring agent."""
    
    # Task configuration
    repo_url: str = Field(..., description="GitHub repository URL")
    repo_path: Optional[str] = Field(None, description="Local path to the repository")
    task_description: str = Field(..., description="User's refactoring task description")
    target_files: Optional[List[str]] = Field(None, description="Specific files to refactor")
    branch_name: str = Field(..., description="Name of the branch to create")
    
    # Planning
    plan: List[RefactoringStep] = Field(default_factory=list, description="List of refactoring steps")
    current_step_index: int = Field(default=0, description="Index of the current step being executed")
    
    # Execution tracking
    processed_files: List[str] = Field(default_factory=list, description="Files already processed")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of actions taken")
    
    # Verification results
    linter_errors: List[str] = Field(default_factory=list, description="Errors from linters")
    test_failures: List[str] = Field(default_factory=list, description="Test failure messages")
    
    # GitHub integration
    pr_title: Optional[str] = Field(None, description="Title for the pull request")
    pr_description: Optional[str] = Field(None, description="Description for the pull request")
    pr_url: Optional[str] = Field(None, description="URL of the created pull request")
    
    # Overall status
    overall_status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall task status")
    error_message: Optional[str] = Field(None, description="Final error message if task failed")
    
    # Control flags
    max_retries: int = Field(default=3, description="Maximum retries per step")
    should_continue: bool = Field(default=True, description="Whether the agent should continue processing")

    class Config:
        arbitrary_types_allowed = True
