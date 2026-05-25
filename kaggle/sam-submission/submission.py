"""SAM Agent - ARC-AGI-3 Kaggle Competition Submission v3.

Fixed for competition mode constraints:
- Single make() per environment (no new episodes)
- Level Reset only (action 0 = restart current level)
- Rate limiting to avoid 429 errors
- Efficient exploration within single session
"""

import subprocess
import sys

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


def solve_game(env, game_id, time_budget=600.0):
    """Solve a game within a SINGLE session (competition mode).
    
    Key constraints:
    - Only Level Resets allowed (action 0 restarts current level)
    - Must solve levels sequentially within one make() call
    - Rate limits on API (add small delays)
    """
    from arcengine import GameAction, GameState

    frame = env.reset()
    if frame is None:
        return 0

    avail = [a for a in frame.available_actions if a != 0]
    
    danger_pairs = set()
    best_level = 0
    current_level = 0
    level_solutions = {}
    deadline = time.monotonic() + time_budget
    total_steps = 0
    
    # Single-session exploration: explore within level, reset on game_over
    path = []
    
    while time.monotonic() < deadline:
        if frame.state == GameState.WIN:
            best_level = max(best_level, frame.levels_completed)
            break
        
        if frame.state == GameState.GAME_OVER:
            # Mark dangerous transitions
            for h, a in path[-5:]:
                danger_pairs.add((h, a))
            
            # Level Reset (goes back to current level start)
            action = GameAction.from_id(0)
            action.action_data.game_id = game_id
            frame = env.step(action, data={})
            total_steps += 1
            if frame is None:
                break
            path = []
            time.sleep(0.05)  # Rate limit protection
            continue
        
        # Track level changes
        if frame.levels_completed > current_level:
            # Solved a level!
            level_solutions[current_level] = path[:]
            logger.info("  %s: level %d solved in %d actions!", game_id, current_level, len(path))
            best_level = max(best_level, frame.levels_completed)
            current_level = frame.levels_completed
            path = []
        
        # Choose action
        h = fast_hash(frame)
        safe = [a for a in avail if (h, a) not in danger_pairs]
        if not safe:
            safe = avail
        
        aid = random.choice(safe)
        
        # Execute action
        action = GameAction.from_id(aid)
        action.action_data.game_id = game_id
        if action.is_complex():
            x, y = random.randint(0, 63), random.randint(0, 63)
            frame = env.step(action, data={"x": x, "y": y})
        else:
            frame = env.step(action, data={})
        
        if frame is None:
            break
        
        total_steps += 1
        path.append((h, aid))
        
        # Periodic rate limit protection
        if total_steps % 100 == 0:
            time.sleep(0.1)
    
    logger.info("  %s: %d levels, %d steps, %d danger pairs",
                game_id, best_level, total_steps, len(danger_pairs))
    return best_level


def main():
    from arc_agi import Arcade, OperationMode

    logger.info("SAM Agent v3 starting...")
    arc = Arcade(operation_mode=OperationMode.COMPETITION)

    environments = arc.get_environments()
    num_envs = len(environments)
    logger.info("Found %d environments", num_envs)

    # Budget: distribute time across games
    # Competition likely has ~2h total, be generous per game
    total_budget = 7200.0  # 2 hours
    time_per_game = total_budget / max(num_envs, 1)
    
    total_levels = 0
    for env_info in environments:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        logger.info("Playing: %s (budget: %.0fs)", game_id, time_per_game)

        try:
            env = arc.make(game_id, save_recording=False)
            if env is None:
                logger.warning("  %s: could not create env", game_id)
                continue
            levels = solve_game(env, game_id, time_budget=time_per_game)
            total_levels += levels
        except Exception as e:
            logger.warning("  %s: error - %s", game_id, e)
        
        time.sleep(0.5)  # Pause between games

    logger.info("DONE: %d total levels across %d games", total_levels, num_envs)


if __name__ == "__main__":
    main()
