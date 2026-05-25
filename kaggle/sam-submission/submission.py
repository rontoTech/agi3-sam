"""SAM Agent - ARC-AGI-3 Kaggle Competition v4.

Fixes:
- Exponential backoff retry on 429 errors
- Much more conservative exploration (fewer steps per game)
- Proper error handling for API rate limits
"""

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "arc-agi>=0.9.8", "-q"])

import hashlib
import random
import time
import logging
import functools

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def retry_on_429(max_retries=5, base_delay=2.0):
    """Decorator to retry on 429 errors with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if "429" in str(e):
                        delay = base_delay * (2 ** attempt) + random.random()
                        logger.info("  Rate limited, waiting %.1fs...", delay)
                        time.sleep(delay)
                    else:
                        raise
            return func(*args, **kwargs)  # Final attempt
        return wrapper
    return decorator


def fast_hash(frame) -> str:
    layer = frame.frame[0]
    if hasattr(layer, "tobytes"):
        return hashlib.md5(layer.tobytes()).hexdigest()[:12]
    return hashlib.md5(str(layer).encode()).hexdigest()[:12]


def safe_step(env, action, data=None):
    """Step with retry on rate limit."""
    for attempt in range(5):
        try:
            result = env.step(action, data=data or {})
            return result
        except Exception as e:
            if "429" in str(e):
                delay = 2.0 * (2 ** attempt) + random.random()
                time.sleep(delay)
            else:
                return None
    return None


def safe_reset(env):
    """Reset with retry on rate limit."""
    for attempt in range(5):
        try:
            result = env.reset()
            return result
        except Exception as e:
            if "429" in str(e):
                delay = 3.0 * (2 ** attempt) + random.random()
                time.sleep(delay)
            else:
                return None
    return None


def solve_game(env, game_id, max_steps=2000):
    """Solve game with conservative step budget and rate limit handling."""
    from arcengine import GameAction, GameState

    frame = safe_reset(env)
    if frame is None:
        return 0

    avail = [a for a in frame.available_actions if a != 0]
    danger_pairs = set()
    best_level = 0
    current_level = 0
    path = []
    steps = 0

    while steps < max_steps:
        if frame.state == GameState.WIN:
            best_level = max(best_level, frame.levels_completed)
            break

        if frame.state == GameState.GAME_OVER:
            for h, a in path[-5:]:
                danger_pairs.add((h, a))
            # Level Reset
            action = GameAction.from_id(0)
            action.action_data.game_id = game_id
            frame = safe_step(env, action)
            steps += 1
            if frame is None:
                break
            path = []
            time.sleep(0.1)
            continue

        if frame.levels_completed > current_level:
            best_level = max(best_level, frame.levels_completed)
            logger.info("  %s: level %d done in %d steps!", game_id, current_level, len(path))
            current_level = frame.levels_completed
            path = []

        h = fast_hash(frame)
        safe = [a for a in avail if (h, a) not in danger_pairs]
        if not safe:
            safe = avail

        aid = random.choice(safe)
        action = GameAction.from_id(aid)
        action.action_data.game_id = game_id
        
        if action.is_complex():
            x, y = random.randint(0, 63), random.randint(0, 63)
            frame = safe_step(env, action, data={"x": x, "y": y})
        else:
            frame = safe_step(env, action)

        if frame is None:
            break

        steps += 1
        path.append((h, aid))
        
        # Rate limit: pause every 50 steps
        if steps % 50 == 0:
            time.sleep(0.2)

    logger.info("  %s: %d levels, %d steps", game_id, best_level, steps)
    return best_level


def main():
    from arc_agi import Arcade, OperationMode

    logger.info("SAM Agent v4 starting...")
    
    arc = Arcade(operation_mode=OperationMode.COMPETITION)
    environments = arc.get_environments()
    num_envs = len(environments)
    logger.info("Found %d environments", num_envs)

    # Conservative step budget per game
    steps_per_game = min(2000, 50000 // max(num_envs, 1))
    
    total_levels = 0
    for i, env_info in enumerate(environments):
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        logger.info("[%d/%d] Playing: %s (%d max steps)", i+1, num_envs, game_id, steps_per_game)

        try:
            env = arc.make(game_id, save_recording=False)
            if env is None:
                continue
            levels = solve_game(env, game_id, max_steps=steps_per_game)
            total_levels += levels
        except Exception as e:
            logger.warning("  %s: error - %s", game_id, e)

        time.sleep(1.0)  # Pause between games

    logger.info("DONE: %d total levels across %d games", total_levels, num_envs)


if __name__ == "__main__":
    main()
