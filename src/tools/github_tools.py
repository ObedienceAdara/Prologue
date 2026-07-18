"""GitHub integration tools for the Codebase Refactor Agent."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from github import Github, GithubException

from .git_auth import build_https_auth, plain_https_url


@dataclass
class GitHubRepoInfo:
    """Information about a GitHub repository."""
    owner: str
    repo_name: str
    full_name: str


def parse_github_url(repo_url: str) -> GitHubRepoInfo:
    """Parse a GitHub URL into owner and repo name.

    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)

    Returns:
        GitHubRepoInfo with parsed information

    Raises:
        ValueError: If the URL is invalid
    """
    if repo_url.startswith("https://github.com/"):
        parts = repo_url.replace("https://github.com/", "").rstrip("/").split("/")
        if len(parts) >= 2:
            repo_name = parts[1].replace(".git", "")
            return GitHubRepoInfo(owner=parts[0], repo_name=repo_name, full_name=f"{parts[0]}/{repo_name}")
    elif repo_url.startswith("git@github.com:"):
        parts = repo_url.replace("git@github.com:", "").rstrip("/").split("/")
        if len(parts) >= 2:
            repo_name = parts[1].replace(".git", "")
            return GitHubRepoInfo(owner=parts[0], repo_name=repo_name, full_name=f"{parts[0]}/{repo_name}")

    raise ValueError(f"Invalid GitHub URL: {repo_url}")


def clone_repository(repo_url: str, target_path: str, token: Optional[str] = None, timeout: int = 300) -> Tuple[bool, str]:
    """Clone a GitHub repository to a local path.

    Security note: unlike the original implementation, the token (if any) is
    never embedded in the clone URL. It's injected as a short-lived HTTP
    Basic Auth header via `git --config-env=...`, which means:
      * it never appears in `git clone`'s argv (not visible via `ps`), and
      * the remote URL git persists to `.git/config` afterwards is the
        plain, tokenless URL - so no credential sits on disk after the run.

    Args:
        repo_url: GitHub repository URL
        target_path: Local path to clone to
        token: GitHub token for authentication (optional for public repos)
        timeout: Timeout in seconds for the clone operation

    Returns:
        Tuple of (success, message)
    """
    try:
        info = parse_github_url(repo_url)
        clone_url = plain_https_url(info.owner, info.repo_name)

        argv = ["git"]
        env = None
        if token:
            extra_args, extra_env = build_https_auth(token)
            argv += extra_args
            import os
            env = {**os.environ, **extra_env}

        argv += ["clone", "--", clone_url, target_path]

        result = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode == 0:
            return True, f"Successfully cloned repository to {target_path}"
        else:
            # Defensive redaction in case an error message ever echoed
            # header contents back (it shouldn't, but cost is free).
            stderr = result.stderr
            if token:
                stderr = stderr.replace(token, "***REDACTED***")
            return False, f"Failed to clone: {stderr}"

    except ValueError as e:
        return False, str(e)
    except subprocess.TimeoutExpired:
        return False, "Clone operation timed out"
    except Exception as e:
        return False, f"Error cloning repository: {str(e)}"


def cleanup_repository(local_path: str) -> Tuple[bool, str]:
    """Remove a locally cloned repository directory.

    The original implementation never cleaned up temp clone directories,
    which left cloned (and potentially token-authenticated, pre-fix)
    repositories sitting on disk indefinitely. Callers should invoke this
    once they're done with a clone unless the user explicitly asked to
    keep it around for inspection.
    """
    try:
        path = Path(local_path).resolve()
        # Refuse to do anything with obviously-wrong paths.
        if str(path) in ("/", str(Path.home())):
            return False, f"Refusing to remove suspicious path: {path}"
        if path.exists():
            shutil.rmtree(path)
        return True, f"Removed {path}"
    except Exception as e:
        return False, f"Error removing {local_path}: {str(e)}"


def create_pull_request(
    repo_url: str,
    branch_name: str,
    title: str,
    description: str,
    base_branch: str = "main",
    token: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """Create a pull request on GitHub.

    Args:
        repo_url: GitHub repository URL
        branch_name: Name of the branch with changes
        title: PR title
        description: PR description
        base_branch: Base branch for the PR (default: main)
        token: GitHub token (required)

    Returns:
        Tuple of (success, message, pr_url)
    """
    if not token:
        return False, "GitHub token is required to create pull requests", None

    try:
        info = parse_github_url(repo_url)
        g = Github(token)
        repo = g.get_repo(info.full_name)

        try:
            repo.get_branch(base_branch)
        except GithubException:
            try:
                base_branch = "master"
                repo.get_branch(base_branch)
            except GithubException:
                # Fall back to whatever the repo actually reports as its default.
                try:
                    base_branch = repo.default_branch
                    repo.get_branch(base_branch)
                except GithubException:
                    return False, "Could not determine the repository's default branch", None

        pr = repo.create_pull(
            title=title,
            body=description,
            head=branch_name,
            base=base_branch
        )

        return True, "Pull request created successfully", pr.html_url

    except GithubException as e:
        return False, f"GitHub API error: {str(e)}", None
    except Exception as e:
        return False, f"Error creating pull request: {str(e)}", None


def get_file_content(
    repo_url: str,
    file_path: str,
    branch: str = "main",
    token: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """Get file content from GitHub."""
    try:
        info = parse_github_url(repo_url)
        g = Github(token) if token else Github()
        repo = g.get_repo(info.full_name)

        contents = repo.get_contents(file_path, ref=branch)

        if isinstance(contents, list):
            return False, "Path is a directory, not a file", None

        import base64
        decoded_content = base64.b64decode(contents.content).decode('utf-8')
        return True, "File retrieved successfully", decoded_content

    except GithubException as e:
        return False, f"GitHub API error: {str(e)}", None
    except Exception as e:
        return False, f"Error getting file content: {str(e)}", None


def update_file_in_repo(
    repo_url: str,
    file_path: str,
    content: str,
    commit_message: str,
    branch: str,
    token: str
) -> Tuple[bool, str]:
    """Update or create a file in a GitHub repository."""
    try:
        info = parse_github_url(repo_url)
        g = Github(token)
        repo = g.get_repo(info.full_name)

        try:
            contents = repo.get_contents(file_path, ref=branch)
            if isinstance(contents, list):
                return False, "Cannot update a directory"

            repo.update_file(
                path=file_path,
                message=commit_message,
                content=content,
                sha=contents.sha,
                branch=branch
            )
        except GithubException:
            repo.create_file(
                path=file_path,
                message=commit_message,
                content=content,
                branch=branch
            )

        return True, f"Successfully updated {file_path} in branch {branch}"

    except GithubException as e:
        return False, f"GitHub API error: {str(e)}"
    except Exception as e:
        return False, f"Error updating file: {str(e)}"
