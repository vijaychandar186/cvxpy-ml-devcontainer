# cvxpy-ml-dev

Convex optimization and machine learning — examples combining [CVXPY](https://www.cvxpy.org/) with PyTorch, NumPy, and Pandas for optimization-driven ML problems.

## Examples

### [`examples/quadratic_optimization/`](examples/quadratic_optimization/)
Quadratic program minimization with linear constraints and sensitivity analysis via dual variables.

### [`examples/team_ranking/`](examples/team_ranking/)
Maximum likelihood team ranking using logistic loss. Trains a skill score vector from pairwise match results and evaluates prediction accuracy.

- `data.py` — training and test match data
- `train.py` — solve the ranking optimization and save `a_hat.npy`
- `evaluate.py` — load saved scores and report ML vs. baseline accuracy

### [`examples/image_colorization/`](examples/image_colorization/)
Image colorization via total variation minimization. Recovers RGB channels from a grayscale image given a sparse set of known pixel colors.

- `prepare_data.py` — generate `flower_given.png` showing known pixels overlaid on grayscale
- `solve.py` — solve the TV minimization problem and save `flower_reconstructed.png`

## Docs

[`docs/cvxpy_notes.md`](docs/cvxpy_notes.md) — reference notes on CVXPY concepts and APIs.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### Dependencies

| Package | Purpose |
|---|---|
| `cvxpy` | Convex optimization modeling |
| `numpy` | Numerical arrays |
| `scipy` | Sparse matrix construction |
| `matplotlib` | Image I/O and plotting |
| `pandas` | Data manipulation |
| `torch` / `torchvision` | Deep learning |
| `ipywidgets` | Jupyter widgets |
| `tqdm` | Progress bars |
| `pillow` | Image processing |
| `fonttools` / `filelock` | Utilities |

## Dev Container

This repo includes a dev container for one-click setup in GitHub Codespaces or VS Code with the Dev Containers extension.

- **Base image:** `mcr.microsoft.com/devcontainers/python:3-3.14-bookworm`
- **Package manager:** uv (via `astral.sh-uv` feature)
- **CPU:** 4 cores
- **On rebuild:** `uv sync --frozen` installs all locked dependencies
- **VS Code extensions:** Jupyter (`ms-toolsai.jupyter`), Python (`ms-python.python`)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vc6698-max/cvxpy)
