---
updated: 2026-11-27T03:15:00+01:00
last_run: { task: T01, result: done }
blocked: false
blocker: ""
next: T02
---
## Last run
- Created `PLAN.md` with an 8-task v0.2 hardening plan derived from `README.md`, `pyproject.toml`, CI workflow, `src/finvigil/`, `evals/`, and `tests/`.
- Created `STATUS.md` and marked `T01` as `done` in `PLAN.md`.
- Checked offline test/lint tooling (`which pytest ruff mypy`, `python3 -m pip list`): no `.venv` exists and `pytest`/`ruff`/`mypy`/project deps are not installed in system Python 3.13.5, so tests/linters could not run offline here.

## Notes for next run
- Sandbox note: `.git` worktree points outside the workspace (`/workspace/team/pm/repos/finvigil/.git/worktrees/...`), so `git` commands fail in sandbox; the runner handles commits if `git commit` fails.
- Start with `T02`: clean up line 28 of `requirements.txt` (`-e /Users/gunnarwrld/proyects/untitled folder`) and enforce non-default `FINVIGIL_AUTH_TOKENS_JSON` in `staging`/`production` in `src/finvigil/config.py` or `src/finvigil/auth.py`.
- If a Python 3.12 `.venv` with `.[dev,prod]` is provisioned in future runs, verify with `ruff check`, `mypy`, `pytest -v`, and `FINVIGIL_LLM_PROVIDER=offline python -m evals.runner`.
