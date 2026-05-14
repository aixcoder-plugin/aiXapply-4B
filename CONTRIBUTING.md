# Contributing

Thanks for your interest in improving `aiXapply`.

## Before You Start

- Read the main `README.md` for the current project layout and reproducibility notes.
- Open an issue before starting large refactors, feature additions, or behavior changes.
- Keep pull requests focused. A small, reviewable PR is much easier to merge than a mixed batch of unrelated changes.

## Development Guidelines

- Do not commit secrets, personal access tokens, `.env` files, checkpoints, datasets, or generated prediction artifacts.
- Prefer repo-relative paths, CLI flags, or environment variables over machine-specific absolute paths.
- Update documentation when changing training entrypoints, evaluation scripts, or environment variables.
- Preserve the existing coding style in touched files. Avoid drive-by formatting changes unrelated to the task.

## Pull Requests

- Describe what changed and why.
- Include the exact commands used for validation when relevant.
- Call out any environment assumptions, such as GPU count, CUDA version, or external model/API dependencies.
- If your change affects evaluation or training behavior, mention whether existing results need to be regenerated.

## Verification

Before opening a PR, run the most relevant checks you can for your change:

- Python syntax checks for edited scripts.
- Targeted unit tests for affected experiment utilities.
- A dry-run or smoke test for training/inference scripts when feasible.
- Documentation review for any changed commands or paths.

## Reporting Issues

When reporting a bug, please include:

- The script or entrypoint you ran.
- The command line or configuration used.
- The observed error message or traceback.
- Enough context to reproduce the problem, including model/provider settings if applicable.

## License

By contributing to this repository, you agree that your contributions will be licensed under the repository's Apache 2.0 license.
