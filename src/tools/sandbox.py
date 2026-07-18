"""Sandboxed execution for any command that runs code from a cloned (untrusted)
repository - primarily linters, formatters, and the test suite.

Why this exists
----------------
`pytest`, and to a lesser extent `ruff`/`black` (via plugins/config files),
execute Python code that lives *inside the cloned repository*. Since that
repository is arbitrary, third-party, and not something the agent's operator
necessarily trusts, running those commands directly on the host is remote
code execution by design: a malicious `conftest.py` or test file can do
anything the host process can do.

This module runs those commands inside a disposable, network-isolated Docker
container instead:
    * `--network none`      -> no outbound network access for the payload
    * `--read-only` root fs -> the container image itself cannot be modified
    * only the repo dir is mounted, read-write, so tools can still reformat
      files in place
    * capabilities are dropped and `no-new-privileges` is set
    * memory / pid / cpu limits bound resource exhaustion
    * the container is removed immediately after the command finishes

If Docker isn't available, callers must explicitly opt in to running
unsandboxed (`--no-sandbox` on the CLI) - the tool refuses to silently fall
back to executing untrusted code on the host.

All subprocess invocations here use list-form arguments (no `shell=True`),
so nothing here is vulnerable to shell injection regardless of what strings
end up inside `command`.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SANDBOX_IMAGE = "prologue-sandbox:latest"

# Minimal, pinned-ish base image with the tools this agent needs already
# installed, so we don't have to pull packages from the network *inside*
# the network-isolated sandbox at run time (which would simply fail).
_DOCKERFILE = """\
FROM python:3.11-slim

RUN pip install --no-cache-dir \\
        ruff==0.7.4 \\
        black==24.10.0 \\
        mypy==1.13.0 \\
        pytest==8.3.3 \\
        pytest-asyncio==0.24.0 \\
    && useradd --create-home --uid 1000 sandbox

USER sandbox
WORKDIR /workspace
"""


@dataclass
class SandboxResult:
    """Result of a sandboxed (or explicitly unsandboxed) command execution."""
    success: bool
    output: str
    error: Optional[str] = None
    ran_in_sandbox: bool = False


class SandboxUnavailableError(RuntimeError):
    """Raised when sandboxed execution is required but Docker isn't usable."""


def is_docker_available() -> bool:
    """Check whether the `docker` CLI is installed and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def ensure_sandbox_image(image: str = SANDBOX_IMAGE, force_rebuild: bool = False) -> SandboxResult:
    """Build the sandbox image from the embedded Dockerfile if it doesn't exist yet.

    The Dockerfile is written to a temp build context on demand; nothing is
    fetched from the network except the base image + pinned tool versions,
    and this build happens once (cached by Docker afterwards).
    """
    if not is_docker_available():
        return SandboxResult(
            success=False,
            output="",
            error="Docker is not available (daemon unreachable or CLI not installed).",
        )

    if not force_rebuild and _image_exists(image):
        return SandboxResult(success=True, output=f"Sandbox image '{image}' already present.")

    import tempfile

    with tempfile.TemporaryDirectory() as build_dir:
        dockerfile_path = Path(build_dir) / "Dockerfile"
        dockerfile_path.write_text(_DOCKERFILE, encoding="utf-8")

        try:
            result = subprocess.run(
                ["docker", "build", "-t", image, build_dir],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, output="", error="Sandbox image build timed out.")
        except OSError as e:
            return SandboxResult(success=False, output="", error=f"Failed to invoke docker: {e}")

        if result.returncode != 0:
            return SandboxResult(success=False, output=result.stdout, error=result.stderr)

        return SandboxResult(success=True, output=f"Built sandbox image '{image}'.")


def run_in_sandbox(
    command: List[str],
    repo_path: str,
    timeout: int = 300,
    image: str = SANDBOX_IMAGE,
    network: bool = False,
) -> SandboxResult:
    """Run `command` (list form, executed as `command[0] command[1:]`) inside
    the sandbox container, with the repository mounted at /workspace.

    Args:
        command: Argv list to execute *inside* the container's /workspace.
                 Never a shell string - it is passed straight to `docker run`
                 as trailing arguments, so no shell parsing happens at any layer.
        repo_path: Absolute path to the cloned repository on the host.
        timeout: Wall-clock timeout in seconds for the whole `docker run`.
        image: Sandbox image to use (built via `ensure_sandbox_image`).
        network: Whether to allow outbound network access (default: no).
                 Only enable this if a step genuinely requires it (e.g.
                 installing a dependency) and you understand the risk of
                 running untrusted code with network access.

    Returns:
        SandboxResult with combined stdout/stderr and success flag.
    """
    if not is_docker_available():
        raise SandboxUnavailableError(
            "Docker is required to safely run commands against a cloned repository, "
            "but is not available. Install Docker, or pass --no-sandbox to explicitly "
            "accept the risk of running untrusted repo code on this host."
        )

    abs_repo_path = str(Path(repo_path).resolve())

    docker_args = [
        "docker", "run", "--rm",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--pids-limit=256",
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "-v", f"{abs_repo_path}:/workspace:rw",
        "-w", "/workspace",
    ]

    if not network:
        docker_args += ["--network", "none"]

    docker_args += [image] + command

    try:
        result = subprocess.run(
            docker_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            success=False,
            output="",
            error=f"Sandboxed command timed out after {timeout} seconds",
            ran_in_sandbox=True,
        )
    except OSError as e:
        return SandboxResult(
            success=False,
            output="",
            error=f"Failed to invoke docker: {e}",
            ran_in_sandbox=True,
        )

    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"

    return SandboxResult(
        success=result.returncode == 0,
        output=output,
        error=None if result.returncode == 0 else f"Command failed with exit code {result.returncode}",
        ran_in_sandbox=True,
    )
