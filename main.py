"""Codebase Refactor Agent - Main CLI entry point."""

import click
import os
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option(
    "--repo",
    "repo_url",
    required=True,
    help="GitHub repository URL (e.g., https://github.com/user/repo)"
)
@click.option(
    "--task",
    "task_description",
    required=True,
    help="Refactoring task description (e.g., 'Add type hints to all functions')"
)
@click.option(
    "--files",
    "target_files",
    default=None,
    help="Comma-separated list of files to refactor (default: all Python files)"
)
@click.option(
    "--branch-name",
    "branch_name",
    default=None,
    help="Name for the refactoring branch (default: auto-generated)"
)
@click.option(
    "--local-path",
    "local_path",
    default=None,
    help="Local path to clone the repository (default: ./refactor-temp-<timestamp>)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run without making actual changes (for testing)"
)
def main(repo_url: str, task_description: str, target_files: str, 
         branch_name: str, local_path: str, dry_run: bool):
    """Codebase Refactor Agent - Autonomous code refactoring with AI.
    
    This tool connects to a GitHub repository, analyzes code for improvements,
    applies refactoring changes, and creates a pull request.
    
    Examples:
    
        # Add type hints to all Python files
        python main.py --repo https://github.com/user/repo --task "Add type hints"
        
        # Refactor specific files
        python main.py --repo https://github.com/user/repo --files src/main.py,src/utils.py --task "Improve code structure"
        
        # Custom branch name
        python main.py --repo https://github.com/user/repo --task "Fix code smells" --branch-name refactor/cleanup
    """
    
    console.print("\n[bold blue]🚀 Codebase Refactor Agent[/bold blue]\n")
    console.print(f"[green]Repository:[/green] {repo_url}")
    console.print(f"[green]Task:[/green] {task_description}")
    
    if target_files:
        console.print(f"[green]Target Files:[/green] {target_files}")
    
    if dry_run:
        console.print("[yellow]⚠️  DRY RUN MODE - No changes will be made[/yellow]\n")
    
    try:
        # Import tools and utilities
        from src.tools import clone_repository, find_python_files, parse_github_url
        from src.nodes import build_refactor_graph
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Validate environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key and not dry_run:
            console.print("[red]Error:[/red] No API key found. Set GROQ_API_KEY in your .env file.")
            sys.exit(1)
        
        # Parse target files
        files_list = None
        if target_files:
            files_list = [f.strip() for f in target_files.split(",")]
        
        # Generate branch name if not provided
        if not branch_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_task = "".join(c if c.isalnum() else "-" for c in task_description[:30]).lower()
            branch_name = f"refactor/{safe_task}-{timestamp}"
        
        # Determine local path
        if not local_path:
            repo_info = parse_github_url(repo_url)
            local_path = f"./refactor-temp-{repo_info.repo_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        console.print(f"[green]Branch Name:[/green] {branch_name}")
        console.print(f"[green]Local Path:[/green] {local_path}\n")
        
        if dry_run:
            console.print("[yellow]📋 Dry run - would perform the following steps:[/yellow]")
            console.print("  1. Clone repository")
            console.print("  2. Analyze codebase")
            console.print("  3. Create refactoring plan")
            console.print("  4. Apply refactoring changes")
            console.print("  5. Run linters and tests")
            console.print("  6. Create branch and commit changes")
            console.print("  7. Open pull request\n")
            console.print("[green]✅ Dry run completed successfully[/green]")
            return
        
        # Step 1: Clone repository
        with console.status("[bold green]Cloning repository..."):
            token = os.getenv("GITHUB_TOKEN")
            success, message = clone_repository(repo_url, local_path, token)
            
            if not success:
                console.print(f"[red]Error cloning repository:[/red] {message}")
                sys.exit(1)
        
        console.print("[green]✓ Repository cloned successfully[/green]\n")
        
        # Step 2: Find Python files
        with console.status("[bold green]Analyzing codebase..."):
            python_files = find_python_files(local_path)
            
        console.print(f"[green]✓ Found {len(python_files)} Python files[/green]\n")
        
        # Step 3: Build and run the agent workflow
        console.print("[bold blue]🤖 Starting AI Agent Workflow...[/bold blue]\n")
        
        # Initialize the LangGraph workflow
        app = build_refactor_graph()
        
        # Initial state
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
            "pr_title": None,
            "pr_description": None,
            "pr_url": None,
            "overall_status": "pending",
            "error_message": None,
            "should_continue": True,
        }
        
        # Run the workflow
        with console.status("[bold green]Running agent workflow..."):
            final_state = app.invoke(initial_state)
        
        # Display results
        console.print("\n[bold blue]📊 Workflow Results[/bold blue]\n")
        
        status = final_state.get("overall_status", "unknown")
        if status == "completed":
            console.print("[green]✓ Refactoring completed successfully![/green]")
            
            if final_state.get("pr_url"):
                console.print(f"\n[bold green]🎉 Pull Request Created:[/bold green]")
                console.print(f"[link={final_state['pr_url']}]{final_state['pr_url']}[/link]")
            else:
                console.print(f"\n[green]Changes committed to branch:[/green] {branch_name}")
                if final_state.get("error_message"):
                    console.print(f"[yellow]Note:[/yellow] {final_state['error_message']}")
                    
        elif status == "failed":
            console.print("[red]✗ Workflow failed[/red]")
            if final_state.get("error_message"):
                console.print(f"[red]Error:[/red] {final_state['error_message']}")
        else:
            console.print(f"[yellow]Workflow status:[/yellow] {status}")
        
        # Summary statistics
        processed_count = len(final_state.get("processed_files", []))
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  • Files processed: {processed_count}")
        console.print(f"  • Steps executed: {final_state.get('current_step_index', 0)}")
        
        if final_state.get("linter_errors"):
            console.print(f"  • Linter issues: {len(final_state['linter_errors'])}")
        if final_state.get("test_failures"):
            console.print(f"  • Test failures: {len(final_state['test_failures'])}")
        
        console.print("\n[green]✅ Done![/green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error:[/red] {str(e)}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
