"""Tool implementations for the Codebase Refactor Agent."""

import os
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


def read_file(file_path: str) -> ToolResult:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        ToolResult with file contents or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path}"
            )
        
        content = path.read_text(encoding='utf-8')
        return ToolResult(success=True, output=content)
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Error reading file: {str(e)}"
        )


def write_file(file_path: str, content: str) -> ToolResult:
    """Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        ToolResult indicating success or failure
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return ToolResult(success=True, output=f"Successfully wrote to {file_path}")
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Error writing file: {str(e)}"
        )


def run_command(command: str, cwd: Optional[str] = None) -> ToolResult:
    """Run a shell command.
    
    Args:
        command: Command to execute
        cwd: Working directory for the command
        
    Returns:
        ToolResult with command output or error
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        
        return ToolResult(
            success=result.returncode == 0,
            output=output,
            error=None if result.returncode == 0 else f"Command failed with exit code {result.returncode}"
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error="Command timed out after 5 minutes"
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Error executing command: {str(e)}"
        )


def run_linter(file_path: str, linter: str = "ruff") -> ToolResult:
    """Run a linter on a file.
    
    Args:
        file_path: Path to the file to lint
        linter: Linter to use (ruff, black, mypy)
        
    Returns:
        ToolResult with linter output
    """
    if linter == "ruff":
        return run_command(f"ruff check {file_path}")
    elif linter == "black":
        return run_command(f"black --check {file_path}")
    elif linter == "mypy":
        return run_command(f"mypy {file_path}")
    else:
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown linter: {linter}"
        )


def run_formatter(file_path: str, formatter: str = "black") -> ToolResult:
    """Run a formatter on a file.
    
    Args:
        file_path: Path to the file to format
        formatter: Formatter to use (black, ruff)
        
    Returns:
        ToolResult with formatter output
    """
    if formatter == "black":
        return run_command(f"black {file_path}")
    elif formatter == "ruff":
        return run_command(f"ruff format {file_path}")
    else:
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown formatter: {formatter}"
        )


def run_tests(test_path: str = ".", cwd: Optional[str] = None) -> ToolResult:
    """Run pytest on the specified path.
    
    Args:
        test_path: Path to tests or specific test file
        cwd: Working directory for running tests
        
    Returns:
        ToolResult with test output
    """
    return run_command(f"pytest {test_path} -v", cwd=cwd)


def find_python_files(directory: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Find all Python files in a directory.
    
    Args:
        directory: Directory to search
        exclude_patterns: Patterns to exclude (e.g., ['__pycache__', '.git'])
        
    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', 'node_modules', '.venv', 'venv']
    
    python_files = []
    dir_path = Path(directory)
    
    for pattern in ['**/*.py']:
        for file_path in dir_path.glob(pattern):
            # Check if any exclude pattern is in the path
            if any(exclude in str(file_path) for exclude in exclude_patterns):
                continue
            python_files.append(str(file_path))
    
    return sorted(python_files)


def get_git_status(cwd: str) -> ToolResult:
    """Get git status of the repository.
    
    Args:
        cwd: Repository root directory
        
    Returns:
        ToolResult with git status output
    """
    return run_command("git status", cwd=cwd)


def create_git_branch(branch_name: str, cwd: str) -> ToolResult:
    """Create and checkout a new git branch.
    
    Args:
        branch_name: Name of the branch to create
        cwd: Repository root directory
        
    Returns:
        ToolResult indicating success or failure
    """
    # First ensure we're on main/master
    result = run_command("git checkout -b {branch_name}", cwd=cwd)
    if not result.success:
        return result
    
    return ToolResult(success=True, output=f"Created and checked out branch: {branch_name}")


def commit_changes(message: str, cwd: str) -> ToolResult:
    """Commit all staged changes.
    
    Args:
        message: Commit message
        cwd: Repository root directory
        
    Returns:
        ToolResult indicating success or failure
    """
    # Stage all changes
    stage_result = run_command("git add -A", cwd=cwd)
    if not stage_result.success:
        return stage_result
    
    # Commit
    return run_command(f'git commit -m "{message}"', cwd=cwd)


def push_branch(branch_name: str, cwd: str) -> ToolResult:
    """Push a branch to remote.
    
    Args:
        branch_name: Name of the branch to push
        cwd: Repository root directory
        
    Returns:
        ToolResult indicating success or failure
    """
    return run_command(f"git push -u origin {branch_name}", cwd=cwd)
