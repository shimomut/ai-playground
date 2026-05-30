# Repository Overview

This repository (`ai-playground`) is a collection of independent AI/LLM learning and experimentation projects. It is intentionally a "playground" — a place to prototype, benchmark, and explore ideas, not a single cohesive product.

## Purpose

- Try out AI/ML/LLM concepts, tools, and infrastructure in small, focused experiments.
- Keep each experiment isolated so failures, dependency conflicts, or environmental quirks in one project do not affect others.
- Preserve historical experiments as a personal reference.

## How projects are organized

- Every experiment lives in its own top-level sub-directory (e.g., `cuda_hello/`, `efa_profiling/`).
- Each project is **self-contained**: it does not import code from sibling projects and does not share build artifacts with them.
- The repository root holds only cross-cutting files (e.g., top-level `README.md`, this steering directory). Avoid adding shared source code or shared dependencies at the root.

## Environment isolation

Every project must run in an isolated environment. Choose one of:

- **Python venv** — preferred for pure Python experiments. Create the venv inside the project directory (e.g., `./.venv`) and add it to the project's `.gitignore`. Use the latest Python interpreter available on `PATH` (resolve via `python3 --version`, or pick the highest `python3.X` found on `PATH`) unless the project explicitly pins an older version for compatibility.
- **Docker** — preferred when the experiment needs system-level dependencies, GPU/CUDA stacks, specific OS libraries, or reproducible images for cluster deployment. Provide a `Dockerfile` in the project directory.

Do not install Python packages into the system or user-global site-packages, and do not rely on a Conda/global interpreter shared across projects. If a project needs both (e.g., a venv for tooling and Docker for runtime), document that in its README.

## Typical project contents

Most projects include some subset of:

- `README.md` — what the project does, how to build/run it, and any prerequisites (GPUs, EFA, specific CUDA versions, AWS access, etc.).
- `Makefile` — primary entry point for build, run, clean, and other common tasks. Prefer `make <target>` over ad-hoc shell commands when a target exists.
- `.gitignore` — project-specific ignores (build outputs, virtual envs, model weights, logs).
- `Dockerfile` — container image definition when the experiment runs in a container.
- Kubernetes manifests (`*.yaml`) — when the experiment runs on a cluster (e.g., EKS, HyperPod).
- Python scripts, CUDA/C++ sources, shell scripts, notebooks — whatever the experiment requires.

## Working in this repo

When adding or modifying a project:

- Keep changes scoped to that project's sub-directory. Do not modify other projects unless explicitly asked.
- If a project has a `Makefile`, use its targets as the source of truth for build/run/test commands.
- If a project has a `README.md`, follow its setup instructions; if it doesn't and you are creating one, add a brief README covering purpose, prerequisites, and how to run.
- Don't introduce a shared dependency mechanism (e.g., a root `requirements.txt`, root `pyproject.toml`, or shared library) without explicit direction. Project independence is a deliberate design choice.

When creating a new project:

- Create a new top-level sub-directory named after the experiment (lowercase, words separated by `_`).
- Add at minimum a `README.md` and a `.gitignore`. Add a `Makefile` when there are repeatable build/run steps.
- Keep the project self-contained: vendor or pin its own dependencies inside the sub-directory.
