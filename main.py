"""Codebase Refactor Agent - Main CLI entry point."""

import click
import os
import sys
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option("--repo", "repo_url", required=True, help="GitHub repository URL (e.g., https://github.com/user/repo)")
@click.option("--task", "task_description", required=True, help="Refactoring task description (e.g., 'Add type hints to all functions')")
@click.option("--files", "target_files", default=None, help="Comma-separated list of files to refactor (default: all Python files)")
@click.option("--branch-name", "branch_name", default=None, help="Name for the refactoring branch (default: auto-generated)")
@click.option("--local-path", "local_path", default=None, help="Local path to clone the repository (default: ./refactor-temp-<timestamp>)")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would happen without cloning or making changes")
@click.option("--yes", "-y", "auto_approve", is_flag=True, default=False, help="Skip the interactive diff review and auto-approve commit/push/PR (use in CI)")
@click.option("--no-sandbox", "no_sandbox", is_flag=True, default=False,
              help="DANGEROUS: run lint/test commands directly on the host instead of in a Docker sandbox. "
                   "Only use this for repositories you fully trust.")
@click.option("--keep-temp", "keep_temp", is_flag=True, default=False, help="Don't delete the local clone directory when the run finishes")
@click.option("--max-retries", "max_retries", default=3, show_default=True, help="Max executor<->verifier retry cycles before giving up")
def main(repo_url: str, task_description: str, target_files: str,
         branch_name: str, local_path: str, dry_run: bool,
         auto_approve: bool, no_sandbox: bool, keep_temp: bool, max_retries: int):
    """Codebase Refactor Agent - AI-assisted code refactoring with human review.

    This tool clones a GitHub repository, plans and applies refactoring
    changes with an LLM, verifies them (sandboxed by default), and then -
    only after you review and approve a diff - commits, pushes, and opens
    a pull request.

    Examples:

        # Add type hints to all Python files
        python main.py --repo https://github.com/user/repo --task "Add type hints"

        # Refactor specific files, auto-approving for CI use
        python main.py --repo https://github.com/user/repo --files src/main.py --task "Improve code structure" --yes
    """

    console.print("\n[bold blue]🚀 Codebase Refactor Agent[/bold blue]\n")
    console.print(f"[green]Repository:[/green] {repo_url}")
    console.print(f"[green]Task:[/green] {task_description}")

    if target_files:
        console.print(f"[green]Target Files:[/green] {target_files}")

    if no_sandbox:
        console.print(
            "[bold red]⚠️  --no-sandbox set: lint/test commands from the cloned repository "
            "will run directly on this host. Only do this for repos you fully trust.[/bold red]\n"
        )

    if dry_run:
        console.print("[yellow]⚠️  DRY RUN MODE - no repository will be cloned and no changes will be made[/yellow]\n")

    from src.tools import clone_repository, cleanup_repository, find_python_files, parse_github_url, get_git_diff, is_docker_available
    from src.nodes import build_refactor_graph, github_integration_node
    from dotenv import load_dotenv

    load_dotenv()

    if not target_files:
        files_list = None
    else:
        files_list = [f.strip() for f in target_files.split(",")]

    if not branch_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_task = "".join(c if c.isalnum() else "-" for c in task_description[:30]).lower().strip("-") or "task"
        branch_name = f"refactor/{safe_task}-{timestamp}"

    try:
        repo_info = parse_github_url(repo_url)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not local_path:
        local_path = f"./refactor-temp-{repo_info.repo_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    console.print(f"[green]Branch Name:[/green] {branch_name}")
    console.print(f"[green]Local Path:[/green] {local_path}\n")

    if dry_run:
        console.print("[yellow]📋 Dry run - would perform the following steps:[/yellow]")
        console.print("  1. Clone repository (credentials injected via header, never embedded in the URL)")
        console.print(f"  2. Verify code changes {'inside a Docker sandbox' if not no_sandbox else '[bold red]directly on the host (--no-sandbox)[/bold red]'}")
        console.print("  3. Create refactoring plan")
        console.print("  4. Apply refactoring changes (syntax-checked before writing)")
        console.print(f"  5. Run linters and tests (bounded to {max_retries} retries)")
        console.print("  6. Show you a diff and ask for approval" + (" (auto-approved: --yes)" if auto_approve else ""))
        console.print("  7. Create branch, commit, push, and open a pull request\n")
        console.print("[green]✅ Dry run completed - nothing was cloned or changed[/green]")
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] No API key found. Set GROQ_API_KEY in your .env file.")
        sys.exit(1)

    if not no_sandbox and not is_docker_available():
        console.print(
            "[red]Error:[/red] Docker is required to safely run lint/test commands against a cloned "
            "repository, but isn't available (daemon unreachable or CLI not installed).\n"
            "Install Docker, or re-run with [bold]--no-sandbox[/bold] to explicitly accept the risk "
            "of running untrusted repository code directly on this host."
        )
        sys.exit(1)

    token = os.getenv("GITHUB_TOKEN")
    cloned = False

    try:
        with console.status("[bold green]Cloning repository..."):
            success, message = clone_repository(repo_url, local_path, token)
            if not success:
                console.print(f"[red]Error cloning repository:[/red] {message}")
                sys.exit(1)
            cloned = True

        console.print("[green]✓ Repository cloned successfully[/green]\n")

        with console.status("[bold green]Analyzing codebase..."):
            python_files = find_python_files(local_path)

        console.print(f"[green]✓ Found {len(python_files)} Python files[/green]\n")

        if not no_sandbox:
            with console.status("[bold green]Preparing sandbox image (first run may take a minute)..."):
                from src.tools import ensure_sandbox_image
                image_result = ensure_sandbox_image()
                if not image_result.success:
                    console.print(f"[red]Error preparing sandbox:[/red] {image_result.error}")
                    sys.exit(1)

        console.print("[bold blue]🤖 Starting AI Agent Workflow...[/bold blue]\n")

        app = build_refactor_graph()

        initial_state = {
            "messages": [],
            "repo_url": repo_url,
            "repo_path": os.path.abspath(local_path),
            "task_description": task_description,
            "target_files": files_list,
            "branch_name": branch_name,
            "plan": [],
            "current_step_index": 0,
            "processed_files": [],
            "execution_history": [],
            "linter_errors": [],
            "test_failures": [],
            "retry_count": 0,
            "max_retries": max_retries,
            "sandbox_enabled": not no_sandbox,
            "pr_title": None,
            "pr_description": None,
            "pr_url": None,
            "overall_status": "pending",
            "error_message": None,
            "should_continue": True,
        }

        with console.status("[bold green]Running agent workflow (plan, edit, verify)..."):
            final_state = app.invoke(initial_state)

        console.print("\n[bold blue]📊 Workflow Results[/bold blue]\n")

        status = final_state.get("overall_status", "unknown")
        processed_count = len(final_state.get("processed_files", []))

        if status == "failed":
            console.print("[red]✗ Workflow failed[/red]")
            if final_state.get("error_message"):
                console.print(f"[red]Error:[/red] {final_state['error_message']}")
            _print_summary(console, final_state)
            return

        if status != "awaiting_approval":
            console.print(f"[yellow]Workflow ended in unexpected status:[/yellow] {status}")
            _print_summary(console, final_state)
            return

        console.print(f"[green]✓ Verification passed.[/green] {processed_count} file(s) changed.\n")

        diff_stat = get_git_diff(final_state["repo_path"], stat_only=True)
        if diff_stat.success and diff_stat.output.strip():
            console.print("[bold]Changes:[/bold]")
            console.print(diff_stat.output)
        else:
            console.print("[yellow]No file changes were produced - nothing to commit.[/yellow]")
            _print_summary(console, final_state)
            return

        approved = auto_approve
        if not approved:
            if not sys.stdin.isatty():
                console.print(
                    "[yellow]Not running interactively and --yes was not passed - stopping before "
                    "commit/push. Re-run with --yes to auto-approve, or inspect the diff at "
                    f"{final_state['repo_path']} yourself.[/yellow]"
                )
                keep_temp = True  # force-keep so there's something to inspect
                _print_summary(console, final_state)
                return

            full_diff = get_git_diff(final_state["repo_path"])
            if full_diff.success:
                console.print("\n[bold]Full diff:[/bold]\n")
                console.print(full_diff.output)
            approved = click.confirm("\nCommit, push, and open a pull request with these changes?", default=False)

        if not approved:
            console.print(
                f"[yellow]Changes declined. They remain uncommitted at {final_state['repo_path']}.[/yellow]"
            )
            keep_temp = True
            _print_summary(console, final_state)
            return

        with console.status("[bold green]Committing, pushing, and opening pull request..."):
            final_state = github_integration_node(final_state, github_token=token)

        status = final_state.get("overall_status", "unknown")
        if status == "completed":
            console.print("[green]✓ Refactoring completed successfully![/green]")
            if final_state.get("pr_url"):
                console.print("\n[bold green]🎉 Pull Request Created:[/bold green]")
                console.print(f"[link={final_state['pr_url']}]{final_state['pr_url']}[/link]")
        elif status == "completed_no_pr":
            console.print("[yellow]✓ Changes committed and pushed, but no PR was opened.[/yellow]")
            if final_state.get("error_message"):
                console.print(f"[yellow]Note:[/yellow] {final_state['error_message']}")
        else:
            console.print("[red]✗ GitHub integration failed[/red]")
            if final_state.get("error_message"):
                console.print(f"[red]Error:[/red] {final_state['error_message']}")

        _print_summary(console, final_state)
        console.print("\n[green]✅ Done![/green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error:[/red] {str(e)}")
        logger.exception("Detailed error:")
        sys.exit(1)
    finally:
        if cloned and not keep_temp:
            ok, msg = cleanup_repository(local_path)
            if not ok:
                console.print(f"[yellow]Warning: could not clean up temp clone: {msg}[/yellow]")
        elif cloned and keep_temp:
            console.print(f"[dim]Local clone kept at: {local_path}[/dim]")


def _print_summary(console: Console, final_state: dict) -> None:
    processed_count = len(final_state.get("processed_files", []))
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  • Files processed: {processed_count}")
    console.print(f"  • Steps executed: {final_state.get('current_step_index', 0)}")
    console.print(f"  • Verification retries used: {final_state.get('retry_count', 0)}")
    if final_state.get("linter_errors"):
        console.print(f"  • Linter issues: {len(final_state['linter_errors'])}")
    if final_state.get("test_failures"):
        console.print(f"  • Test failures: {len(final_state['test_failures'])}")


if __name__ == "__main__":
    main()
