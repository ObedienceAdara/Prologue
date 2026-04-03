"""Basic tests for the Codebase Refactor Agent tools."""

import pytest
from pathlib import Path
import tempfile
import os


class TestFileTools:
    """Tests for file operation tools."""
    
    def test_read_file_success(self):
        """Test reading an existing file."""
        from src.tools import read_file
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name
        
        try:
            result = read_file(temp_path)
            assert result.success is True
            assert result.output == "Hello, World!"
            assert result.error is None
        finally:
            os.unlink(temp_path)
    
    def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        from src.tools import read_file
        
        result = read_file("/nonexistent/path/file.txt")
        assert result.success is False
        assert "File not found" in result.error
    
    def test_write_file_success(self):
        """Test writing to a file."""
        from src.tools import write_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.txt")
            result = write_file(file_path, "Test content")
            
            assert result.success is True
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == "Test content"
    
    def test_write_file_creates_directories(self):
        """Test that write_file creates parent directories."""
        from src.tools import write_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nested", "dir", "test.txt")
            result = write_file(file_path, "Test content")
            
            assert result.success is True
            assert Path(file_path).exists()


class TestRunCommand:
    """Tests for command execution tool."""
    
    def test_run_command_success(self):
        """Test running a successful command."""
        from src.tools import run_command
        
        result = run_command("echo 'Hello'")
        assert result.success is True
        assert "Hello" in result.output
    
    def test_run_command_failure(self):
        """Test running a failing command."""
        from src.tools import run_command
        
        result = run_command("exit 1")
        assert result.success is False
        assert result.error is not None


class TestFindPythonFiles:
    """Tests for finding Python files."""
    
    def test_find_python_files(self):
        """Test finding Python files in a directory."""
        from src.tools import find_python_files
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some Python files
            Path(temp_dir, "file1.py").touch()
            Path(temp_dir, "file2.py").touch()
            Path(temp_dir, "not_python.txt").touch()
            pycache_dir = Path(temp_dir, "__pycache__")
            pycache_dir.mkdir()
            (pycache_dir / "cached.py").touch()
            
            files = find_python_files(temp_dir)
            
            assert len(files) == 2
            assert all(f.endswith('.py') for f in files)
            assert not any('__pycache__' in f for f in files)


class TestGitHubURLParsing:
    """Tests for GitHub URL parsing."""
    
    def test_parse_https_url(self):
        """Test parsing HTTPS GitHub URL."""
        from src.tools import parse_github_url
        
        info = parse_github_url("https://github.com/owner/repo")
        assert info.owner == "owner"
        assert info.repo_name == "repo"
        assert info.full_name == "owner/repo"
    
    def test_parse_https_url_with_git_extension(self):
        """Test parsing HTTPS URL with .git extension."""
        from src.tools import parse_github_url
        
        info = parse_github_url("https://github.com/owner/repo.git")
        assert info.owner == "owner"
        assert info.repo_name == "repo"
    
    def test_parse_ssh_url(self):
        """Test parsing SSH GitHub URL."""
        from src.tools import parse_github_url
        
        info = parse_github_url("git@github.com:owner/repo.git")
        assert info.owner == "owner"
        assert info.repo_name == "repo"
    
    def test_parse_invalid_url(self):
        """Test parsing invalid URL raises error."""
        from src.tools import parse_github_url
        
        with pytest.raises(ValueError):
            parse_github_url("https://gitlab.com/owner/repo")


class TestAgentState:
    """Tests for state models."""
    
    def test_agent_state_creation(self):
        """Test creating an AgentState instance."""
        from src.state import AgentState, TaskStatus
        
        state = AgentState(
            repo_url="https://github.com/test/repo",
            task_description="Add type hints",
            branch_name="refactor/type-hints"
        )
        
        assert state.repo_url == "https://github.com/test/repo"
        assert state.task_description == "Add type hints"
        assert state.overall_status == TaskStatus.PENDING
        assert state.plan == []
    
    def test_refactoring_step_creation(self):
        """Test creating a RefactoringStep instance."""
        from src.state import RefactoringStep, TaskStatus
        
        step = RefactoringStep(
            step_id=1,
            description="Add type hints to main.py",
            file_path="src/main.py",
            action="add_type_hints"
        )
        
        assert step.step_id == 1
        assert step.status == TaskStatus.PENDING
        assert step.retry_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
