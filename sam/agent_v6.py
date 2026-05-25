"""SAM Agent v6 - ARC-AGI-3 with Status Bar Masking + Hierarchical Exploration.

KEY INSIGHT from 3rd place solution: The frame includes UI elements (step counter,
strike counter, score display) that change every step. This means the SAME logical
game state gets DIFFERENT hashes, exploding the state space.

FIX: Detect and mask the status bar region before hashing. This makes identical
logical states hash to the same value, dramatically reducing the effective state space.

Detection strategy:
1. The status bar is typically at the top or bottom of the frame
2. It contains rapidly changing pixels (counter changes every step)
3. It's a horizontal strip (full width, few rows tall)

Additional improvements:
- Hierarchical action priority (segment frame for click targets)
- Connected component analysis for smarter exploration
- Navigate to nearest frontier (minimize path to untested actions)
"""

import hashlib
import random
import time
import logging
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def detect_status_bar(frames_history: list, rows_to_check: int = 8) -> tuple[int, int]:
    """Detect status bar region by finding rows that change frequently.
    
    Returns (mask_top_rows, mask_bottom_rows) — number of rows to mask from top/bottom.
    """
    if len(frames_history) < 3:
        return 0, 0
    
    # Compare consecutive frames to find rows that change
    top_changes = [0] * rows_to_check
    bottom_changes = [0] * rows_to_check
    
    for i in range(1, min(len(frames_history), 10)):
        f1 = frames_history[i-1]
        f2 = frames_history[i]
        if f1 is None or f2 is None:
            continue
        
        g1 = f1.frame[0]
        g2 = f2.frame[0]
        if hasattr(g1, 'tolist'):
            g1 = g1.tolist()
        if hasattr(g2, 'tolist'):
            g2 = g2.tolist()
        
        h = min(len(g1), len(g2))
        if h < rows_to_check * 2:
            continue
        
        # Check top rows
        for row in range(min(rows_to_check, h)):
            if g1[row] != g2[row]:
                top_changes[row] += 1
        
        # Check bottom rows
        for row in range(min(rows_to_check, h)):
            row_idx = h - 1 - row
            if g1[row_idx] != g2[row_idx]:
                bottom_changes[row] += 1
    
    # Mask rows that change in > 50% of frame pairs
    threshold = max(1, len(frames_history) // 4)
    mask_top = 0
    for i in range(rows_to_check):
        if top_changes[i] >= threshold:
            mask_top = i + 1
        else:
            break
    
    mask_bottom = 0
    for i in range(rows_to_check):
        if bottom_changes[i] >= threshold:
            mask_bottom = i + 1
        else:
            break
    
    return mask_top, mask_bottom


def masked_hash(frame, mask_top: int = 0, mask_bottom: int = 0) -> str:
    """Hash frame with status bar masked out."""
    layer = frame.frame[0]
    if hasattr(layer, 'tolist'):
        layer_list = layer.tolist()
    else:
        layer_list = layer
    
    h = len(layer_list)
    if mask_top > 0 or mask_bottom > 0:
        end = h - mask_bottom if mask_bottom > 0 else h
        start = mask_top
        if start < end:
            masked = layer_list[start:end]
        else:
            masked = layer_list
    else:
        masked = layer_list
    
    raw = str(masked).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def segment_frame(frame) -> list[dict]:
    """Segment frame into connected components for click prioritization.
    
    Returns list of segments with: color, size, center_x, center_y, is_button_like
    """
    layer = frame.frame[0]
    if hasattr(layer, 'tolist'):
        grid = layer.tolist()
    else:
        grid = layer
    
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return []
    
    # Quick color histogram for dominant colors
    color_counts = {}
    for y in range(0, h, 4):  # Sample every 4th row for speed
        for x in range(0, w, 4):
            c = grid[y][x]
            color_counts[c] = color_counts.get(c, 0) + 1
    
    # Background is the most common color
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find non-background regions (simplified segmentation)
    segments = []
    visited = set()
    
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if grid[y][x] != bg_color and (y, x) not in visited:
                # BFS to find connected region
                color = grid[y][x]
                queue = deque([(y, x)])
                region = []
                while queue and len(region) < 100:
                    cy, cx = queue.popleft()
                    if (cy, cx) in visited:
                        continue
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if grid[cy][cx] != color:
                        continue
                    visited.add((cy, cx))
                    region.append((cy, cx))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        queue.append((cy+dy, cx+dx))
                
                if len(region) >= 4:
                    avg_y = sum(p[0] for p in region) / len(region)
                    avg_x = sum(p[1] for p in region) / len(region)
                    segments.append({
                        'color': color,
                        'size': len(region),
                        'center_x': int(avg_x),
                        'center_y': int(avg_y),
                        'is_button': 4 <= len(region) <= 50,
                    })
    
    return segments


def get_click_priorities(segments: list[dict]) -> list[tuple[int, int, float]]:
    """Get prioritized click coordinates from segments.
    
    Returns list of (x, y, priority) sorted by priority (highest first).
    """
    candidates = []
    for seg in segments:
        if seg['is_button']:
            priority = 5.0  # Button-like segments have highest priority
        elif seg['size'] > 50:
            priority = 2.0  # Large regions
        else:
            priority = 1.0  # Small decorative elements
        candidates.append((seg['center_x'], seg['center_y'], priority))
    
    candidates.sort(key=lambda x: -x[2])
    return candidates


class SAMAgentV6:
    """Competition agent with status bar masking and hierarchical exploration."""

    def __init__(self):
        self.mask_top = 0
        self.mask_bottom = 0
        self.frame_history = []
        self.danger_pairs = set()
        self.graph = {}  # hash -> {action -> (next_hash, worked)}
        self.click_priorities = []
        self.click_idx = 0

    def solve_game(self, env, game_id, max_steps=2000):
        """Solve game with masking and hierarchical exploration."""
        from arcengine import GameAction, GameState

        frame = env.reset()
        if frame is None:
            return 0

        avail = [a for a in frame.available_actions if a != 0]
        has_click = 6 in avail
        
        self.frame_history = [frame]
        self.danger_pairs = set()
        self.graph = {}
        
        best_level = 0
        current_level = 0
        path = []
        steps = 0
        calibration_steps = 0

        while steps < max_steps:
            if frame.state == GameState.WIN:
                best_level = max(best_level, frame.levels_completed)
                break

            if frame.state == GameState.GAME_OVER:
                for h, a in path[-5:]:
                    self.danger_pairs.add((h, a))
                action = GameAction.from_id(0)
                action.action_data.game_id = game_id
                frame = env.step(action, data={})
                steps += 1
                if frame is None:
                    break
                path = []
                self.frame_history.append(frame)
                continue

            if frame.levels_completed > current_level:
                best_level = max(best_level, frame.levels_completed)
                logger.info("  %s: level %d solved in %d steps!", game_id, current_level, len(path))
                current_level = frame.levels_completed
                path = []
                self.frame_history = [frame]
                self.mask_top = 0
                self.mask_bottom = 0
                calibration_steps = 0

            # Calibrate status bar masking from first few steps
            if calibration_steps < 5 and len(self.frame_history) >= 3:
                self.mask_top, self.mask_bottom = detect_status_bar(self.frame_history)
                calibration_steps += 1

            # Hash with masking
            h = masked_hash(frame, self.mask_top, self.mask_bottom)
            
            if h not in self.graph:
                self.graph[h] = {}
                # Get click priorities for this frame
                if has_click:
                    segments = segment_frame(frame)
                    self.click_priorities = get_click_priorities(segments)
                    self.click_idx = 0

            node = self.graph[h]
            
            # Choose action with hierarchical priority
            aid, click_x, click_y = self._choose_action(h, avail, has_click)
            
            # Execute
            action = GameAction.from_id(aid)
            action.action_data.game_id = game_id
            if action.is_complex():
                frame = env.step(action, data={"x": click_x, "y": click_y})
            else:
                frame = env.step(action, data={})
            
            if frame is None:
                break
            
            steps += 1
            path.append((h, aid))
            self.frame_history.append(frame)
            
            # Record transition
            new_h = masked_hash(frame, self.mask_top, self.mask_bottom)
            changed = (new_h != h)
            node[aid] = (new_h, changed)

        logger.info("  %s: %d levels, %d steps, mask=(%d,%d), graph=%d",
                    game_id, best_level, steps, self.mask_top, self.mask_bottom, len(self.graph))
        return best_level

    def _choose_action(self, h: str, avail: list[int], has_click: bool) -> tuple[int, int, int]:
        """Hierarchical action selection."""
        node = self.graph.get(h, {})
        
        # Priority 1: Untested actions that aren't dangerous
        untested = [a for a in avail if a not in node and (h, a) not in self.danger_pairs]
        
        if untested:
            # Prefer keyboard actions (more likely to be productive)
            keyboard_untested = [a for a in untested if a <= 5]
            if keyboard_untested:
                aid = random.choice(keyboard_untested)
                return aid, 32, 32
            
            # Click action with priority targeting
            if 6 in untested and self.click_priorities:
                if self.click_idx < len(self.click_priorities):
                    x, y, _ = self.click_priorities[self.click_idx]
                    self.click_idx += 1
                    return 6, x, y
                return 6, random.randint(0, 63), random.randint(0, 63)
            
            return random.choice(untested), 32, 32
        
        # Priority 2: Actions that previously caused state changes
        changing = [a for a in avail if a in node and node[a][1] and (h, a) not in self.danger_pairs]
        if changing:
            aid = random.choice(changing)
            if aid == 6:
                return aid, random.randint(0, 63), random.randint(0, 63)
            return aid, 32, 32
        
        # Priority 3: Any non-dangerous action
        safe = [a for a in avail if (h, a) not in self.danger_pairs]
        if safe:
            aid = random.choice(safe)
            if aid == 6:
                return aid, random.randint(0, 63), random.randint(0, 63)
            return aid, 32, 32
        
        # Last resort
        aid = random.choice(avail)
        return aid, 32, 32
