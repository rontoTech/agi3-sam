"""SAM Agent v7 - Full competition agent with frontier navigation.

Combines all best practices:
1. Status bar masking (reduce state space)
2. Frontier navigation via BFS (navigate to nearest untested state)
3. Hierarchical action priority (keyboard > buttons > random clicks)
4. Danger avoidance (learn from GAME_OVER)
5. Path optimization (find shorter solutions on retry)
6. Rate limit handling (retry with backoff)

This is the algorithm from the 3rd place paper, implemented cleanly:
- Build directed state graph during play
- Track tested/untested actions per state
- When current state fully explored, BFS to nearest frontier
- Navigate to frontier via known edges
- Try untested action from frontier
- Repeat until level solved or budget exhausted
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "arc-agi>=0.9.8", "-q"])

import hashlib, random, time, logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def with_retry(func, retries=5, delay=3.0):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = delay * (2 ** attempt) + random.random() * 2
            time.sleep(wait)


def detect_mask(frames, rows=8):
    """Detect status bar rows to mask."""
    if len(frames) < 3:
        return 0, 0
    top, bot = [0]*rows, [0]*rows
    for i in range(1, min(len(frames), 8)):
        g1 = frames[i-1].frame[0]
        g2 = frames[i].frame[0]
        if hasattr(g1, 'tolist'): g1 = g1.tolist()
        if hasattr(g2, 'tolist'): g2 = g2.tolist()
        h = min(len(g1), len(g2))
        for r in range(min(rows, h)):
            if g1[r] != g2[r]: top[r] += 1
            if g1[h-1-r] != g2[h-1-r]: bot[r] += 1
    t = max(1, len(frames) // 4)
    mt = sum(1 for i in range(rows) if top[i] >= t and (i == 0 or top[i-1] >= t))
    mb = sum(1 for i in range(rows) if bot[i] >= t and (i == 0 or bot[i-1] >= t))
    return mt, mb


def mhash(frame, mt=0, mb=0):
    """Hash frame with masking."""
    layer = frame.frame[0]
    if hasattr(layer, 'tobytes'):
        data = layer.tobytes()
        # Mask by zeroing top/bottom rows worth of bytes
        if mt > 0 or mb > 0:
            if hasattr(layer, 'tolist'):
                layer = layer.tolist()
            else:
                layer = list(layer)
            h = len(layer)
            start = mt
            end = h - mb if mb > 0 else h
            if start >= end: start, end = 0, h
            raw = str(layer[start:end]).encode()
            return hashlib.md5(raw).hexdigest()[:12]
        return hashlib.md5(data).hexdigest()[:12]
    if hasattr(layer, 'tolist'): layer = layer.tolist()
    h = len(layer)
    start = mt
    end = h - mb if mb > 0 else h
    if start >= end: start, end = 0, h
    return hashlib.md5(str(layer[start:end]).encode()).hexdigest()[:12]


class StateGraph:
    """Directed state graph for frontier navigation."""
    
    def __init__(self):
        self.nodes = {}  # hash -> {action -> next_hash}
        self.tested = {}  # hash -> set of tested actions
        self.danger = set()  # (hash, action) pairs that lead toward GAME_OVER
    
    def add_state(self, h):
        if h not in self.nodes:
            self.nodes[h] = {}
            self.tested[h] = set()
    
    def add_edge(self, from_h, action, to_h):
        self.add_state(from_h)
        self.add_state(to_h)
        self.nodes[from_h][action] = to_h
        self.tested[from_h].add(action)
    
    def mark_tested(self, h, action):
        self.add_state(h)
        self.tested[h].add(action)
    
    def untested_actions(self, h, avail):
        """Get untested, non-dangerous actions from state h."""
        tested = self.tested.get(h, set())
        return [a for a in avail if a not in tested and (h, a) not in self.danger]
    
    def find_frontier_path(self, start_h, avail):
        """BFS to find shortest path to a state with untested actions."""
        if not start_h or start_h not in self.nodes:
            return None
        
        queue = deque([(start_h, [])])
        visited = {start_h}
        
        while queue:
            h, path = queue.popleft()
            if len(path) > 100:
                continue
            
            # Check if this state has untested actions
            if path and self.untested_actions(h, avail):
                return path
            
            # Expand via known edges
            for action, next_h in self.nodes.get(h, {}).items():
                if next_h not in visited and (h, action) not in self.danger:
                    visited.add(next_h)
                    queue.append((next_h, path + [action]))
        
        return None


def solve_game(env, game_id, max_steps=2000):
    """Solve game with frontier navigation."""
    from arcengine import GameAction, GameState

    frame = with_retry(env.reset)
    if frame is None:
        return 0

    avail = [a for a in frame.available_actions if a != 0]
    has_click = 6 in avail
    
    graph = StateGraph()
    frames_hist = [frame]
    mt, mb = 0, 0
    calibrated = False
    
    best_level = 0
    current_level = 0
    path_actions = []  # actions taken since last reset/level-up
    steps = 0
    nav_queue = []  # navigation actions to replay

    while steps < max_steps:
        if frame.state == GameState.WIN:
            best_level = max(best_level, frame.levels_completed)
            break

        if frame.state == GameState.GAME_OVER:
            # Mark last actions as dangerous
            h_cur = mhash(frame, mt, mb) if frames_hist else ""
            recent_path = []
            # Reconstruct state-action pairs from path_actions
            # (simplified: mark current hash as dangerous for last action)
            for ha_pair in list(zip(
                [mhash(f, mt, mb) for f in frames_hist[-6:-1]] if len(frames_hist) > 5 else [],
                path_actions[-5:]
            )):
                graph.danger.add(ha_pair)
            
            # Level Reset
            action = GameAction.from_id(0)
            action.action_data.game_id = game_id
            frame = with_retry(lambda: env.step(action, data={}))
            steps += 1
            if frame is None: break
            path_actions = []
            nav_queue = []
            frames_hist = [frame]
            continue

        if frame.levels_completed > current_level:
            best_level = max(best_level, frame.levels_completed)
            logger.info("  %s: L%d solved (%d steps)!", game_id, current_level, len(path_actions))
            current_level = frame.levels_completed
            path_actions = []
            nav_queue = []
            frames_hist = [frame]
            graph = StateGraph()  # Fresh graph for new level
            mt, mb = 0, 0
            calibrated = False

        # Calibrate masking
        if not calibrated and len(frames_hist) >= 4:
            mt, mb = detect_mask(frames_hist)
            calibrated = True

        h = mhash(frame, mt, mb)
        graph.add_state(h)

        # If navigating to frontier, follow the nav queue
        if nav_queue:
            aid = nav_queue.pop(0)
        else:
            # Try untested actions first
            untested = graph.untested_actions(h, avail)
            
            if untested:
                # Prefer keyboard actions
                kb_untested = [a for a in untested if a <= 5]
                if kb_untested:
                    aid = random.choice(kb_untested)
                elif 6 in untested:
                    aid = 6
                else:
                    aid = random.choice(untested)
            else:
                # All actions tested — find frontier via BFS
                frontier_path = graph.find_frontier_path(h, avail)
                if frontier_path:
                    nav_queue = frontier_path[1:]  # Rest of path
                    aid = frontier_path[0]
                else:
                    # No reachable frontier — try random
                    safe = [a for a in avail if (h, a) not in graph.danger]
                    aid = random.choice(safe) if safe else random.choice(avail)

        # Execute action
        graph.mark_tested(h, aid)
        action = GameAction.from_id(aid)
        action.action_data.game_id = game_id
        
        if action.is_complex():
            cx, cy = random.randint(0, 63), random.randint(0, 63)
            frame = with_retry(lambda: env.step(action, data={"x": cx, "y": cy}))
        else:
            frame = with_retry(lambda: env.step(action, data={}))
        
        if frame is None: break
        steps += 1
        path_actions.append(aid)
        frames_hist.append(frame)
        
        # Record transition in graph
        new_h = mhash(frame, mt, mb)
        if new_h != h:
            graph.add_edge(h, aid, new_h)

    logger.info("  %s: %d levels, %d steps, %d states, mask(%d,%d)",
                game_id, best_level, steps, len(graph.nodes), mt, mb)
    return best_level


def main():
    from arc_agi import Arcade, OperationMode

    logger.info("SAM Agent v7 starting...")
    arc = Arcade(operation_mode=OperationMode.COMPETITION)
    environments = with_retry(arc.get_environments)
    n = len(environments)
    logger.info("Found %d environments", n)

    budget = min(2000, 40000 // max(n, 1))
    total = 0

    for i, ei in enumerate(environments):
        gid = ei.game_id if hasattr(ei, 'game_id') else str(ei)
        logger.info("[%d/%d] %s (%d steps)", i+1, n, gid, budget)
        try:
            env = with_retry(lambda: arc.make(gid, save_recording=False))
            if env is None: continue
            levels = solve_game(env, gid, max_steps=budget)
            total += levels
        except Exception as e:
            logger.warning("  %s: %s", gid, str(e)[:80])
        time.sleep(1.0)

    logger.info("DONE: %d levels across %d games", total, n)


if __name__ == "__main__":
    main()
