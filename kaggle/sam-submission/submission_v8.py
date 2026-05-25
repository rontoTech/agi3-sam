"""SAM Agent v8 - Optimized for Kaggle competition constraints.

Lessons learned from v1-v7:
1. Rate limits: ~2-5 requests/second to ARC API
2. Each env.step() is ONE API call
3. With 25 games, need to be very strategic about step budget
4. Status bar masking reduces state space
5. Click games need segmentation (can't try all 4096 positions)
6. Danger avoidance prevents GAME_OVER waste

v8 Strategy:
- 0.3s delay between EVERY API call (max ~3 calls/sec)
- Early termination if game seems unsolvable (no progress in N steps)  
- Adaptive budget: start small, extend if making progress
- Smart click: try center first, then grid pattern (not random)
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "arc-agi>=0.9.8", "-q"])

import hashlib, random, time, logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

API_DELAY = 0.3  # seconds between each API call


def with_retry(func, retries=5, delay=5.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = delay * (2 ** attempt) + random.random() * 3
            logger.info("  retry %d: %.0fs wait (%s)", attempt+1, wait, str(e)[:50])
            time.sleep(wait)


def mhash(frame, mt=0, mb=0):
    layer = frame.frame[0]
    if hasattr(layer, 'tolist'): layer = layer.tolist()
    h = len(layer)
    s, e = mt, h - mb if mb > 0 else h
    if s >= e: s, e = 0, h
    return hashlib.md5(str(layer[s:e]).encode()).hexdigest()[:12]


def detect_mask(frames, rows=6):
    if len(frames) < 3: return 0, 0
    top, bot = [0]*rows, [0]*rows
    for i in range(1, min(len(frames), 6)):
        g1, g2 = frames[i-1].frame[0], frames[i].frame[0]
        if hasattr(g1, 'tolist'): g1 = g1.tolist()
        if hasattr(g2, 'tolist'): g2 = g2.tolist()
        h = min(len(g1), len(g2))
        for r in range(min(rows, h)):
            if g1[r] != g2[r]: top[r] += 1
            if g1[h-1-r] != g2[h-1-r]: bot[r] += 1
    t = max(1, len(frames) // 3)
    mt = sum(1 for i in range(rows) if top[i] >= t and (i == 0 or top[i-1] >= t))
    mb = sum(1 for i in range(rows) if bot[i] >= t and (i == 0 or bot[i-1] >= t))
    return mt, mb


def get_click_grid(step_num):
    """Generate systematic click positions (not random).
    Grid pattern covers the frame methodically."""
    positions = []
    # Center first
    positions.append((32, 32))
    # 8-point star
    for x, y in [(16,16),(48,16),(16,48),(48,48),(32,16),(32,48),(16,32),(48,32)]:
        positions.append((x, y))
    # 4x4 grid
    for gx in range(4):
        for gy in range(4):
            positions.append((8 + gx*16, 8 + gy*16))
    # More detail
    for gx in range(8):
        for gy in range(8):
            positions.append((4 + gx*8, 4 + gy*8))
    idx = step_num % len(positions)
    return positions[idx]


def solve_game(env, game_id, initial_budget=500, max_budget=1500):
    """Solve with adaptive budget and rate limiting."""
    from arcengine import GameAction, GameState

    time.sleep(API_DELAY)
    frame = with_retry(env.reset)
    if frame is None: return 0

    avail = [a for a in frame.available_actions if a != 0]
    has_click = 6 in avail
    
    danger = set()
    graph = {}  # hash -> {action -> next_hash}
    tested = {}  # hash -> set(actions)
    fhist = [frame]
    mt, mb = 0, 0
    calibrated = False
    
    best_level = 0
    current_level = 0
    path = []
    steps = 0
    steps_since_progress = 0
    budget = initial_budget
    click_step = 0

    while steps < budget:
        time.sleep(API_DELAY)
        
        if frame.state == GameState.WIN:
            best_level = max(best_level, frame.levels_completed)
            break

        if frame.state == GameState.GAME_OVER:
            for i in range(max(0, len(fhist)-6), len(fhist)-1):
                h2 = mhash(fhist[i], mt, mb) if i < len(fhist) else ""
                if h2 and i-max(0,len(fhist)-6) < len(path):
                    danger.add((h2, path[i-max(0,len(fhist)-6)]))
            action = GameAction.from_id(0)
            action.action_data.game_id = game_id
            frame = with_retry(lambda: env.step(action, data={}))
            steps += 1
            if frame is None: break
            path = []; fhist = [frame]
            continue

        if frame.levels_completed > current_level:
            best_level = max(best_level, frame.levels_completed)
            logger.info("  %s: L%d solved! (%d steps)", game_id, current_level, steps)
            current_level = frame.levels_completed
            path = []; fhist = [frame]; graph = {}; tested = {}
            mt, mb = 0, 0; calibrated = False
            steps_since_progress = 0
            budget = max_budget  # Extend budget on progress!

        # Early termination if stuck
        steps_since_progress += 1
        if steps_since_progress > initial_budget and best_level == 0:
            logger.info("  %s: giving up (no progress in %d steps)", game_id, steps_since_progress)
            break

        # Calibrate masking
        if not calibrated and len(fhist) >= 4:
            mt, mb = detect_mask(fhist)
            calibrated = True

        h = mhash(frame, mt, mb)
        if h not in graph: graph[h] = {}
        if h not in tested: tested[h] = set()

        # Choose action (hierarchical)
        untested_safe = [a for a in avail if a not in tested[h] and (h, a) not in danger]
        
        if untested_safe:
            # Prefer keyboard over click
            kb = [a for a in untested_safe if a <= 5]
            if kb:
                aid = random.choice(kb)
                cx, cy = 32, 32
            elif 6 in untested_safe:
                aid = 6
                cx, cy = get_click_grid(click_step)
                click_step += 1
            else:
                aid = random.choice(untested_safe)
                cx, cy = 32, 32
        else:
            # BFS to frontier
            frontier_path = None
            q = deque([(h, [])]); vis = {h}
            while q:
                nh, p = q.popleft()
                if len(p) > 50: continue
                if p and [a for a in avail if a not in tested.get(nh, set()) and (nh, a) not in danger]:
                    frontier_path = p; break
                for a2, nxt in graph.get(nh, {}).items():
                    if nxt not in vis and (nh, a2) not in danger:
                        vis.add(nxt); q.append((nxt, p+[a2]))
            
            if frontier_path:
                aid = frontier_path[0]
                cx, cy = 32, 32
            else:
                safe = [a for a in avail if (h, a) not in danger]
                aid = random.choice(safe) if safe else random.choice(avail)
                cx, cy = random.randint(0, 63), random.randint(0, 63) if aid == 6 else (32, 32)

        # Execute
        tested[h].add(aid)
        action = GameAction.from_id(aid)
        action.action_data.game_id = game_id
        if action.is_complex():
            frame = with_retry(lambda: env.step(action, data={"x": cx, "y": cy}))
        else:
            frame = with_retry(lambda: env.step(action, data={}))
        if frame is None: break
        
        steps += 1
        path.append(aid)
        fhist.append(frame)
        new_h = mhash(frame, mt, mb)
        if new_h != h:
            graph[h][aid] = new_h

    logger.info("  %s: %d levels, %d steps, %d states", game_id, best_level, steps, len(graph))
    return best_level


def main():
    from arc_agi import Arcade, OperationMode

    logger.info("SAM v8 - Rate-optimized competition agent")
    arc = Arcade(operation_mode=OperationMode.COMPETITION)
    
    time.sleep(API_DELAY)
    environments = with_retry(arc.get_environments)
    n = len(environments)
    logger.info("Found %d environments", n)

    total = 0
    for i, ei in enumerate(environments):
        gid = ei.game_id if hasattr(ei, 'game_id') else str(ei)
        logger.info("[%d/%d] %s", i+1, n, gid)
        try:
            time.sleep(1.0)
            env = with_retry(lambda: arc.make(gid, save_recording=False))
            if env is None: continue
            levels = solve_game(env, gid, initial_budget=500, max_budget=1500)
            total += levels
        except Exception as e:
            logger.warning("  %s: FAIL - %s", gid, str(e)[:60])
        time.sleep(2.0)

    logger.info("TOTAL: %d levels across %d games", total, n)


if __name__ == "__main__":
    main()
