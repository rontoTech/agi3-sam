#!/usr/bin/env python3
"""Run the competition agent on ARC-AGI-3 games.

Usage:
    python scripts/run_competition.py --game ls20-9607627b
    python scripts/run_competition.py --all --time-limit 60
    python scripts/run_competition.py --game ls20-9607627b --use-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.config import find_metadata
from sam.explorer.online_explorer import OnlineGraphExplorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("competition")


def find_all_games(env_dir: str) -> list[str]:
    """Find all game IDs in the environments directory."""
    games = []
    env_path = Path(env_dir)
    for meta_path in sorted(env_path.rglob("metadata.json")):
        try:
            data = json.loads(meta_path.read_text())
            game_id = data.get("game_id", "")
            if game_id:
                games.append(game_id)
        except (json.JSONDecodeError, KeyError):
            continue
    return games


def run_single_game(
    game_id: str,
    env_dir: str,
    cache_dir: Path | None = None,
) -> dict:
    """Run online graph explorer on a single game."""
    cache_path = None
    if cache_dir:
        cache_path = cache_dir / f"{game_id.replace('-', '_')}_solutions.json"

    agent = OnlineGraphExplorer(
        game_id=game_id,
        environments_dir=env_dir,
        solutions_cache=cache_path,
    )

    t0 = time.time()
    try:
        result = agent.solve()
        result["time_s"] = round(time.time() - t0, 1)
        return result
    except Exception as e:
        logger.error("Error on %s: %s", game_id, e, exc_info=True)
        return {
            "game_id": game_id,
            "win": False,
            "levels_completed": 0,
            "error": str(e),
            "time_s": round(time.time() - t0, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Competition Agent")
    parser.add_argument("--game", default=None, help="Single game ID to solve")
    parser.add_argument("--env-dir", default="environment_files")
    parser.add_argument("--all", action="store_true", help="Run on all games")
    parser.add_argument("--use-cache", action="store_true", help="Use/save solution cache")
    parser.add_argument("--output", default="sam/runs/competition_results.json")
    args = parser.parse_args()

    cache_dir = Path("sam/checkpoints/solutions") if args.use_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        games = find_all_games(args.env_dir)
        logger.info("Found %d games", len(games))
    elif args.game:
        games = [args.game]
    else:
        parser.error("Specify --game GAME_ID or --all")
        return

    results = []
    total_score = 0.0
    total_possible = 0.0
    wins = 0

    for game_id in games:
        logger.info("=" * 60)
        logger.info("Running: %s", game_id)
        result = run_single_game(game_id, args.env_dir, cache_dir)
        results.append(result)

        if result.get("win"):
            wins += 1
        rhae = result.get("rhae", {})
        score = rhae.get("total_score", 0)
        max_score = rhae.get("max_possible", 0)
        total_score += score
        total_possible += max_score

        logger.info(
            "Result: win=%s levels=%s steps=%s RHAE=%.1f/%.1f time=%.1fs",
            result.get("win"),
            result.get("levels_completed"),
            result.get("total_env_steps", 0),
            score,
            max_score,
            result.get("time_s", 0),
        )

    logger.info("=" * 60)
    logger.info(
        "SUMMARY: %d/%d wins, RHAE %.1f/%.1f (%.1f%%)",
        wins, len(games), total_score, total_possible,
        (total_score / total_possible * 100) if total_possible > 0 else 0,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "summary": {
            "games": len(games),
            "wins": wins,
            "total_rhae": total_score,
            "max_rhae": total_possible,
            "percentage": (total_score / total_possible * 100) if total_possible > 0 else 0,
        },
        "results": results,
    }, indent=2))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
