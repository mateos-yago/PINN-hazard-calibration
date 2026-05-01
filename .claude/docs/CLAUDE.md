# CLAUDE.md — Project Guidelines

## Architecture

- Use an object-oriented design; split logic into modules with clear, single responsibilities.
- Keep documentation short and purposeful — one-line docstrings where possible, no multi-paragraph blocks.

## Neural Networks

- Use **PyTorch** for all neural network code.
- Vectorize or tensorize operations wherever possible; avoid explicit Python loops over tensors.
- Keep code readable — no cryptic one-liners.

## Experiments & Results

- Every experiment must be logged with its hyperparameters.
- Save trained model weights per experiment so any run can be reloaded at any time.
- Export all results; organize output directories in a logical, consistent structure.
- Include a brief comment in the experiment config or log explaining the rationale behind that run.

## Version Control

- Commit and push to the GitHub remote together — don't leave commits local-only.
- Commit often with clean, descriptive messages so any version can be recovered easily.
