"""GitHub integration tools for the Codebase Refactor Agent."""

import os
from typing import Optional, Tuple
from github import Github, GithubException
from dataclasses import dataclass


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
    # Handle various URL formats
    if repo_url.startswith("https://github.com/"):
        parts = repo_url.replace("https://github.com/", "").rstrip("/").split("/")
        if len(parts) >= 2:
            repo_name = parts[1].replace(".git", "")
            return GitHubRepoInfo(
                owner=parts[0],
                repo_name=repo_name,
                full_name=f"{parts[0]}/{repo_name}"
            )
    elif repo_url.startswith("git@github.com:"):
        parts = repo_url.replace("git@github.com:", "").rstrip("/").split("/")
        if len(parts) >= 2:
            repo_name = parts[1].replace(".git", "")
            return GitHubRepoInfo(
                owner=parts[0],
                repo_name=repo_name,
                full_name=f"{parts[0]}/{repo_name}"
            )
    
    raise ValueError(f"Invalid GitHub URL: {repo_url}")


def clone_repository(repo_url: str, target_path: str, token: Optional[str] = None) -> Tuple[bool, str]:
    """Clone a GitHub repository to a local path.
    
    Args:
        repo_url: GitHub repository URL
        target_path: Local path to clone to
        token: GitHub token for authentication (optional for public repos)
        
    Returns:
        Tuple of (success, message)
    """
    import subprocess
    
    try:
        # Construct authenticated URL if token is provided
        if token:
            info = parse_github_url(repo_url)
            auth_url = f"https://{token}@github.com/{info.full_name}.git"
        else:
            auth_url = repo_url
            if not auth_url.endswith(".git"):
                auth_url += ".git"
        
        result = subprocess.run(
            ["git", "clone", auth_url, target_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True, f"Successfully cloned repository to {target_path}"
        else:
            return False, f"Failed to clone: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Clone operation timed out"
    except Exception as e:
        return False, f"Error cloning repository: {str(e)}"


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
        # Parse repo info
        info = parse_github_url(repo_url)
        
        # Initialize GitHub client
        g = Github(token)
        repo = g.get_repo(info.full_name)
        
        # Check if base branch exists, try master if main doesn't exist
        try:
            repo.get_branch(base_branch)
        except GithubException:
            try:
                base_branch = "master"
                repo.get_branch(base_branch)
            except GithubException:
                return False, f"Neither 'main' nor 'master' branch found", None
        
        # Create the pull request
        pr = repo.create_pull(
            title=title,
            body=description,
            head=branch_name,
            base=base_branch
        )
        
        return True, f"Pull request created successfully", pr.html_url
        
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
    """Get file content from GitHub.
    
    Args:
        repo_url: GitHub repository URL
        file_path: Path to the file in the repo
        branch: Branch to get the file from
        token: GitHub token (may be required for private repos)
        
    Returns:
        Tuple of (success, content_or_error, decoded_content)
    """
    try:
        info = parse_github_url(repo_url)
        g = Github(token) if token else Github()
        repo = g.get_repo(info.full_name)
        
        contents = repo.get_contents(file_path, ref=branch)
        
        # Handle both single file and directory responses
        if isinstance(contents, list):
            return False, "Path is a directory, not a file", None
        
        # Decode content
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
    """Update or create a file in a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL
        file_path: Path to the file
        content: New content for the file
        commit_message: Commit message
        branch: Branch to update
        token: GitHub token (required)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        info = parse_github_url(repo_url)
        g = Github(token)
        repo = g.get_repo(info.full_name)
        
        # Try to get existing file
        try:
            contents = repo.get_contents(file_path, ref=branch)
            if isinstance(contents, list):
                return False, "Cannot update a directory"
            
            # Update existing file
            repo.update_file(
                path=file_path,
                message=commit_message,
                content=content,
                sha=contents.sha,
                branch=branch
            )
        except GithubException:
            # File doesn't exist, create it
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
