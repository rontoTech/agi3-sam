"""SAM Agent - ARC-AGI-3 Kaggle Competition Submission."""

import subprocess
import sys

# Install arc-agi package
subprocess.check_call([sys.executable, "-m", "pip", "install", "arc-agi>=0.9.8", "-q"])

import hashlib
import random
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def fast_hash(frame) -> str:
    layer = frame.frame[0]
    if hasattr(layer, "tobytes"):
        return hashlib.md5(layer.tobytes()).hexdigest()[:12]
    return hashlib.md5(str(layer).encode()).hexdigest()[:12]


def solve_game(env, game_id, time_budget=7200.0):
    """Solve a game using fast exploration with danger avoidance."""
    from arcengine import GameAction, GameState

    frame = env.reset()
    if frame is None:
        return 0

    avail = [a for a in frame.available_actions if a != 0]

    danger_pairs = set()
    solutions = {}
    best_level = 0
    deadline = time.monotonic() + time_budget

    episodes = 0
    while time.monotonic() < deadline:
        episodes += 1

        # Level Reset
        action = GameAction.from_id(0)
        action.action_data.game_id = game_id
        frame = env.step(action, data={})
        if frame is None:
            continue

        # Replay known solutions to reach frontier
        current_level = 0
        replay_ok = True
        for lvl in sorted(solutions.keys()):
            if lvl != current_level:
                break
            for aid in solutions[lvl]:
                action = GameAction.from_id(aid)
                action.action_data.game_id = game_id
                if action.is_complex():
                    frame = env.step(action, data={"x": 32, "y": 32})
                else:
                    frame = env.step(action, data={})
                if frame is None or frame.state == GameState.GAME_OVER:
                    replay_ok = False
                    break
            if not replay_ok:
                break
            current_level = frame.levels_completed

        if not replay_ok or frame is None:
            continue

        # Explore from frontier
        path = []
        for step in range(1000):
            if frame.state == GameState.WIN:
                break
            if frame.state == GameState.GAME_OVER:
                for h, a in path[-5:]:
                    danger_pairs.add((h, a))
                break

            h = fast_hash(frame)
            safe = [a for a in avail if (h, a) not in danger_pairs]
            if not safe:
                safe = avail

            aid = random.choice(safe)

            action = GameAction.from_id(aid)
            action.action_data.game_id = game_id
            if action.is_complex():
                x, y = random.randint(0, 63), random.randint(0, 63)
                frame = env.step(action, data={"x": x, "y": y})
            else:
                frame = env.step(action, data={})

            if frame is None:
                break
            path.append((h, aid))

            if frame.levels_completed > current_level:
                sol = [a for _, a in path]
                if current_level not in solutions or len(sol) < len(solutions[current_level]):
                    solutions[current_level] = sol
                    logger.info(
                        "  %s level %d: %d actions (ep %d)",
                        game_id, current_level, len(sol), episodes
                    )
                best_level = max(best_level, frame.levels_completed)
                current_level = frame.levels_completed
                path = []

    logger.info(
        "  %s: %d levels, %d episodes, %d danger pairs",
        game_id, best_level, episodes, len(danger_pairs)
    )
    return best_level


def main():
    from arc_agi import Arcade, OperationMode

    logger.info("SAM Agent starting...")
    arc = Arcade(operation_mode=OperationMode.COMPETITION)

    environments = arc.get_environments()
    num_envs = len(environments)
    logger.info("Found %d environments", num_envs)

    total_levels = 0
    # Distribute time budget across games (2 hour total typical)
    time_per_game = min(7200.0 / max(num_envs, 1), 1800.0)

    for env_info in environments:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        logger.info("Playing: %s (budget: %.0fs)", game_id, time_per_game)

        try:
            env = arc.make(game_id, save_recording=False)
            if env is None:
                continue
            levels = solve_game(env, game_id, time_budget=time_per_game)
            total_levels += levels
        except Exception as e:
            logger.warning("  %s error: %s", game_id, e)

    logger.info("DONE: %d total levels across %d games", total_levels, num_envs)


if __name__ == "__main__":
    main()
