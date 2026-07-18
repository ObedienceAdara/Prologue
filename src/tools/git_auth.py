"""Credential handling for git-over-HTTPS operations.

The original implementation embedded the GitHub token directly into the
clone/push URL (`https://<token>@github.com/...`). That has two problems:

1. git persists the remote URL - including the embedded token - into
   `.git/config` in plaintext, where it sits indefinitely on disk.
2. The token becomes part of a subprocess argument list, which is visible
   to any other process on the same host via `ps aux` / `/proc/<pid>/cmdline`
   for as long as the git process runs.

This module instead injects the credential as an HTTP Basic Auth header via
git's `--config-env=<key>=<envvar>` mechanism (git >= 2.31). The header value
lives only in an environment variable for the lifetime of the single git
subprocess call - it is never written to disk, never appears in argv, and
the remote URL git stores is the plain, tokenless URL.
"""

from __future__ import annotations

import base64
from typing import Dict, List, Tuple

_ENV_VAR_NAME = "PROLOGUE_GIT_HTTP_AUTH_HEADER"


def build_https_auth(token: str) -> Tuple[List[str], Dict[str, str]]:
    """Build the extra argv + env needed to authenticate a single git HTTPS call.

    Usage:
        extra_args, extra_env = build_https_auth(token)
        run_command(["git", *extra_args, "clone", plain_url, dest], env=extra_env)

    Args:
        token: GitHub personal access token / installation token.

    Returns:
        (extra_args, extra_env) - extra_args must be inserted immediately
        after "git" and before the subcommand; extra_env must be merged
        into the subprocess environment.
    """
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    header_value = f"AUTHORIZATION: basic {basic}"
    extra_args = [f"--config-env=http.extraHeader={_ENV_VAR_NAME}"]
    extra_env = {_ENV_VAR_NAME: header_value}
    return extra_args, extra_env


def plain_https_url(owner: str, repo_name: str) -> str:
    """The tokenless clone/push URL that will actually be persisted to .git/config."""
    return f"https://github.com/{owner}/{repo_name}.git"
