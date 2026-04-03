# Codebase Refactor Agent (Legacy Cleaner)

An autonomous AI agent that connects to a GitHub repository, analyzes code for "code smells", creates a refactoring plan, applies changes in a new branch, runs tests, and opens a Pull Request.

## Features

- **Perception**: Reads code from file system or GitHub API
- **Reasoning/Planning**: Creates multi-step refactoring plans
- **Tool Use**: Integrates with git, linters (ruff, black), pytest, and GitHub API
- **Memory**: Tracks processed files to avoid infinite loops
- **Self-Correction**: Analyzes linter/test errors and attempts alternative fixes

## Architecture

Built with:
- **Language**: Python 3.10+
- **Agent Framework**: LangGraph (explicit state and cycle control)
- **LLM Interface**: OpenAI API or Anthropic Claude
- **State Management**: Pydantic models for type-safe state tracking

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- `GITHUB_TOKEN` (for PR creation)

## Usage

```bash
# Add type hints to a repository
python main.py --repo https://github.com/user/repo --task "Add type hints"

# Refactor specific files
python main.py --repo https://github.com/user/repo --files src/main.py,src/utils.py --task "Improve code structure"

# Run with custom branch name
python main.py --repo https://github.com/user/repo --task "Fix code smells" --branch-name refactor/cleanup
```

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── tools/          # Tool implementations (file ops, git, linters)
│   ├── nodes/          # Agent nodes (planner, executor, verifier)
│   └── state/          # State models and management
├── tests/              # Test suite
├── main.py             # CLI entry point
├── requirements.txt    # Dependencies
└── .env.example        # Environment template
```

## How It Works

1. **Planner Node**: Analyzes the request and breaks it into executable steps
2. **Executor Node**: Applies code changes using available tools
3. **Verifier Node**: Runs linters and tests; loops back if issues found
4. **GitHub Integration**: Creates branch, commits changes, opens PR with summary

## Example Workflow

```
User: "Add type hints to all Python files"
  ↓
Planner: Creates plan [1. Scan files, 2. Add imports, 3. Add annotations]
  ↓
Executor: Processes each file sequentially
  ↓
Verifier: Runs ruff/mypy → If errors, send back to Executor
  ↓
Success: Create branch, commit, open PR
```

## License

MIT