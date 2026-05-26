#!/usr/bin/env python3
"""SAM v2 runner: pretrain + run on ARC-AGI-3 games.

Usage:
    # Pretrain on all 25 games
    python scripts/run_sam_v2.py --pretrain --epochs 10
    
    # Run on a single game with pretrained checkpoint
    python scripts/run_sam_v2.py --game ls20-9607627b
    
    # Run on all games  
    python scripts/run_sam_v2.py --all
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sam_v2")


def pretrain(args) -> None:
    """Run pretraining pipeline."""
    from sam.train.pretrain_v2 import pretrain_sam
    
    logger.info("Starting SAM v2 pretraining...")
    report = pretrain_sam(
        config_path=Path(args.config),
        output_path=Path(args.checkpoint),
        epochs=args.epochs,
        steps_per_game=args.steps_per_game,
        episodes_per_game=args.episodes_per_game,
        batch_size=args.batch_size,
        device=args.device,
    )
    logger.info("Pretraining complete:")
    print(json.dumps(report, indent=2))


def run_game(game_id: str, args) -> dict:
    """Run SAM v2 on a single game."""
    from sam.agent_v2 import SamV2Agent
    
    ckpt = Path(args.checkpoint) if Path(args.checkpoint).exists() else None
    
    agent = SamV2Agent(
        game_id=game_id,
        environments_dir=args.env_dir,
        checkpoint=ckpt,
        planning_ms=args.planning_ms,
        num_simulations=args.simulations,
        device=args.device,
    )
    
    t0 = time.time()
    try:
        result = agent.run()
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


def find_all_games(env_dir: str) -> list[str]:
    """Find all game IDs."""
    games = []
    for meta_path in sorted(Path(env_dir).rglob("metadata.json")):
        try:
            data = json.loads(meta_path.read_text())
            if "game_id" in data:
                games.append(data["game_id"])
        except Exception:
            continue
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM v2 Agent")
    parser.add_argument("--pretrain", action="store_true", help="Run pretraining")
    parser.add_argument("--game", default=None, help="Single game ID")
    parser.add_argument("--all", action="store_true", help="Run on all games")
    parser.add_argument("--env-dir", default="environment_files")
    parser.add_argument("--config", default="sam/configs/global.yaml")
    parser.add_argument("--checkpoint", default="sam/checkpoints/global_sam.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--planning-ms", type=float, default=50.0)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-game", type=int, default=500)
    parser.add_argument("--episodes-per-game", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="sam/runs/sam_v2_results.json")
    args = parser.parse_args()

    if args.pretrain:
        pretrain(args)
        return

    if args.all:
        games = find_all_games(args.env_dir)
    elif args.game:
        games = [args.game]
    else:
        parser.error("Specify --pretrain, --game GAME_ID, or --all")
        return

    results = []
    total_score = 0.0
    total_possible = 0.0
    wins = 0

    for game_id in games:
        logger.info("=" * 60)
        logger.info("Running SAM v2 on: %s", game_id)
        result = run_game(game_id, args)
        results.append(result)

        if result.get("win"):
            wins += 1
        rhae = result.get("rhae", {})
        score = rhae.get("total_score", 0)
        max_score = rhae.get("max_possible", 0)
        total_score += score
        total_possible += max_score

        logger.info(
            "  Result: win=%s levels=%s steps=%s RHAE=%.1f/%.1f plan=%d explore=%d wm_err=%.4f",
            result.get("win"),
            result.get("levels_completed"),
            result.get("env_steps", 0),
            score, max_score,
            result.get("planning_actions", 0),
            result.get("explore_actions", 0),
            result.get("avg_wm_error", 0),
        )

    logger.info("=" * 60)
    pct = (total_score / total_possible * 100) if total_possible > 0 else 0
    logger.info(
        "FINAL: %d/%d wins | RHAE %.1f/%.1f (%.1f%%)",
        wins, len(games), total_score, total_possible, pct,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "summary": {
            "games": len(games),
            "wins": wins,
            "total_rhae": total_score,
            "max_rhae": total_possible,
            "percentage": pct,
        },
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
