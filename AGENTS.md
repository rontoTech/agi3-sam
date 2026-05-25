# AGENTS.md

## Cursor Cloud specific instructions

### Environment

- Python 3.12 with a virtualenv at `.venv/`. Activate with `source .venv/bin/activate`.
- No Docker, no databases, no external services required — everything runs offline.
- PyTorch runs on CPU (no GPU needed; model is ~1M params).

### Key commands

| Task | Command |
|------|---------|
| Run tests | `python -m pytest tests/ -q` |
| Run SAM agent | `python scripts/run_sam.py --game ls20-9607627b --env-dir environment_files` |
| BFS solver (single level) | `python scripts/solve_ls20.py --level 0 --time-limit 30` |
| BFS solver (all levels) | `python scripts/solve_ls20.py --all --time-limit 900` |

### Notes

- The SAM agent currently completes **level 1 only** (levels 2+ require cached BFS plans). This is expected behavior per `CLOUD_HANDOFF.md`.
- The BFS solver outputs plans to `sam/checkpoints/ls20_plans.json`. Solving all 7 levels is CPU-intensive and may take up to 15 minutes.
- Game environment files live in `environment_files/` and are already committed to the repo — no download needed.
- See `sam/DESIGN.md` for architecture details and `CLOUD_HANDOFF.md` for the full development continuation prompt.
