"""Online graph exploration agent for ARC-AGI-3.

Key insight: Build a directed state graph during live play.
Use RESET + replay to navigate to frontier nodes (states with untried actions).
BFS over the graph finds shortest paths to the best exploration targets.

Algorithm:
1. For levels with known solutions → replay them (maximum RHAE efficiency)
2. For unknown levels → systematic online exploration:
   a. From current state, try untried actions (skip no-ops)
   b. Record transitions in state graph
   c. When stuck (all actions tried) → RESET + replay to nearest frontier
   d. frontier = nearest node with untried actions reachable from root
3. When level-up detected → extract shortest path via graph BFS

No neural network needed. Pure systematic graph exploration.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc_agi import Arcade, OperationMode
from arc_agi.wrapper import EnvironmentWrapper
from arcengine import FrameDataRaw, GameAction, GameState

from sam.config import GameMetadata, find_metadata
from sam.explorer import fast_frame_hash, frame_visual_diff

logger = logging.getLogger(__name__)

DEFAULT_LS20_CACHE = Path("sam/checkpoints/ls20_plans.json")


@dataclass
class GraphNode:
    """Node in the online state graph."""
    hash: str
    levels_completed: int
    available_actions: list[int]
    tried_actions: set[int] = field(default_factory=set)
    edges: dict[int, str] = field(default_factory=dict)
    noop_actions: set[int] = field(default_factory=set)
    depth: int = 0


class OnlineGraphExplorer:
    """Online graph-based exploration agent.
    
    Uses pre-computed solutions where available, and systematic
    graph exploration for unsolved levels.
    """

    def __init__(
        self,
        game_id: str,
        environments_dir: str = "environment_files",
        solutions_cache: Path | None = None,
    ) -> None:
        self.game_id = game_id
        self.environments_dir = environments_dir
        self.metadata = find_metadata(game_id, Path(environments_dir))
        self.solutions_cache = solutions_cache
        
        self.graph: dict[str, GraphNode] = {}
        self.root_hash: str | None = None
        self.current_path: list[int] = []
        self.level_solutions: dict[int, list[int]] = {}
        self.total_steps = 0
        
        self._load_all_caches()

    def _load_all_caches(self) -> None:
        """Load from both custom cache and the default ls20 plans."""
        if self.solutions_cache and self.solutions_cache.exists():
            self._load_json_cache(self.solutions_cache)
        
        if DEFAULT_LS20_CACHE.exists() and "ls20" in self.game_id:
            self._load_json_cache(DEFAULT_LS20_CACHE)

    def _load_json_cache(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for k, v in data.get("level_plans", {}).items():
                level = int(k)
                actions = list(v)
                if level not in self.level_solutions or len(actions) < len(self.level_solutions[level]):
                    self.level_solutions[level] = actions
        except Exception as e:
            logger.debug("Cache load error: %s", e)

    def _save_cache(self) -> None:
        """Save discovered solutions."""
        if not self.solutions_cache:
            return
        self.solutions_cache.parent.mkdir(parents=True, exist_ok=True)
        full_path = []
        for i in sorted(self.level_solutions.keys()):
            full_path.extend(self.level_solutions[i])
        self.solutions_cache.write_text(json.dumps({
            "game_id": self.game_id,
            "level_plans": {str(k): v for k, v in sorted(self.level_solutions.items())},
            "full_path": full_path,
        }, indent=2))

    def _get_node(self, frame: FrameDataRaw, h: str, depth: int = 0) -> GraphNode:
        if h not in self.graph:
            self.graph[h] = GraphNode(
                hash=h,
                levels_completed=frame.levels_completed,
                available_actions=[a for a in frame.available_actions if a != 0],
                depth=depth,
            )
        return self.graph[h]

    def _budget(self, level: int) -> int:
        if self.metadata:
            return self.metadata.max_actions_for_level(level)
        return 1000

    def _step_env(self, env: EnvironmentWrapper, action_id: int) -> FrameDataRaw | None:
        """Execute action in environment."""
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = self.game_id
        elif action.is_complex():
            action.set_data({"game_id": self.game_id, "x": 32, "y": 32})
        frame = env.step(action)
        if frame is not None:
            self.total_steps += 1
        return frame

    def solve(self) -> dict[str, Any]:
        """Main solve loop."""
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=self.environments_dir,
        )
        env = arcade.make(self.game_id, save_recording=False)
        if env is None:
            raise RuntimeError(f"Cannot create env for {self.game_id}")

        frame = env.reset()
        if frame is None:
            raise RuntimeError("Reset failed")

        level_action_counts: list[int] = []

        while frame.state != GameState.WIN:
            level = frame.levels_completed
            budget = self._budget(level)

            if level in self.level_solutions:
                frame, steps = self._replay_solution(env, frame, level)
                if frame is None:
                    break
                if frame.levels_completed > level:
                    level_action_counts.append(steps)
                    logger.info("Level %d replayed in %d steps (RHAE %.1f)",
                        level, steps,
                        min((self.metadata.baseline_for_level(level) / steps) ** 2 * 100, 115.0) if self.metadata else 0
                    )
                    continue
                else:
                    logger.warning("Replay failed for level %d, falling back to exploration", level)
                    del self.level_solutions[level]

            frame, steps, solved = self._explore_level(env, frame, budget)
            if frame is None:
                break
            if solved:
                level_action_counts.append(steps)
            else:
                logger.warning("Failed to solve level %d after %d steps", level, steps)
                break

        win = frame is not None and frame.state == GameState.WIN
        self._save_cache()
        return self._build_result(
            win,
            frame.levels_completed if frame else 0,
            level_action_counts,
        )

    def _replay_solution(
        self, env: EnvironmentWrapper, frame: FrameDataRaw, level: int
    ) -> tuple[FrameDataRaw | None, int]:
        """Replay a known solution path."""
        actions = self.level_solutions[level]
        steps = 0
        for aid in actions:
            frame = self._step_env(env, aid)
            if frame is None:
                return None, steps
            steps += 1
            if frame.state == GameState.GAME_OVER:
                return frame, steps
            if frame.levels_completed > level:
                return frame, steps
        return frame, steps

    def _explore_level(
        self, env: EnvironmentWrapper, frame: FrameDataRaw, budget: int
    ) -> tuple[FrameDataRaw | None, int, bool]:
        """Explore current level using online graph BFS.
        
        Strategy:
        1. Try untried productive actions from current state
        2. When stuck, RESET and replay to nearest frontier
        3. Detect level-up and extract shortest path
        """
        level = frame.levels_completed
        
        self.graph = {}
        start_hash = fast_frame_hash(frame)
        self.root_hash = start_hash
        self._get_node(frame, start_hash, depth=0)
        self.current_path = []
        
        steps = 0
        replay_queue: list[int] = []
        consecutive_noops = 0
        
        while steps < budget:
            if frame.state == GameState.WIN:
                break

            if frame.state == GameState.GAME_OVER:
                frame = self._step_env(env, GameAction.RESET.value)
                if frame is None:
                    return None, steps, False
                steps += 1
                self.current_path = []
                replay_queue = []
                continue

            if replay_queue:
                aid = replay_queue.pop(0)
                prev_hash = fast_frame_hash(frame)
                frame = self._step_env(env, aid)
                if frame is None:
                    return None, steps, False
                steps += 1
                self.current_path.append(aid)
                
                new_hash = fast_frame_hash(frame)
                if new_hash != prev_hash:
                    self._get_node(frame, new_hash, depth=len(self.current_path))
                
                if frame.levels_completed > level:
                    self._record_solution(level)
                    return frame, steps, True
                continue

            cur_hash = fast_frame_hash(frame)
            node = self._get_node(frame, cur_hash, depth=len(self.current_path))
            
            productive_untried = [
                a for a in node.available_actions
                if a not in node.tried_actions and a not in node.noop_actions
            ]
            
            if productive_untried:
                aid = self._pick_action(productive_untried, node, frame)
                node.tried_actions.add(aid)
                
                prev_hash = cur_hash
                frame = self._step_env(env, aid)
                if frame is None:
                    return None, steps, False
                steps += 1
                
                new_hash = fast_frame_hash(frame)
                
                if new_hash == prev_hash:
                    node.noop_actions.add(aid)
                    consecutive_noops += 1
                else:
                    node.edges[aid] = new_hash
                    self.current_path.append(aid)
                    new_node = self._get_node(frame, new_hash, depth=len(self.current_path))
                    consecutive_noops = 0
                    
                    if frame.levels_completed > level:
                        self._record_solution(level)
                        return frame, steps, True
                    
                    if frame.state == GameState.GAME_OVER:
                        continue
            else:
                frontier_path = self._find_frontier_path(level)
                if frontier_path is None:
                    logger.info(
                        "Level %d: no frontier (explored %d states, %d steps used)",
                        level, len(self.graph), steps
                    )
                    return frame, steps, False
                
                frame = self._step_env(env, GameAction.RESET.value)
                if frame is None:
                    return None, steps, False
                steps += 1
                self.current_path = []
                replay_queue = list(frontier_path)

        return frame, steps, False

    def _record_solution(self, level: int) -> None:
        """Record the solution path and try to optimize it via graph BFS."""
        raw_solution = self.current_path[:]
        
        opt = self._extract_shortest_path(level)
        if opt and len(opt) < len(raw_solution):
            solution = opt
        else:
            solution = raw_solution
        
        self.level_solutions[level] = solution
        logger.info(
            "Level %d solved! Raw path: %d, Optimized: %d",
            level, len(raw_solution), len(solution)
        )

    def _extract_shortest_path(self, target_level: int) -> list[int] | None:
        """BFS from root to find shortest path to level-up in known graph."""
        if self.root_hash is None:
            return None
        
        queue: deque[tuple[str, list[int]]] = deque([(self.root_hash, [])])
        visited = {self.root_hash}
        
        while queue:
            node_hash, path = queue.popleft()
            node = self.graph.get(node_hash)
            if not node:
                continue
            if node.levels_completed > target_level:
                return path
            for aid, next_hash in node.edges.items():
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((next_hash, path + [aid]))
        
        return None

    def _find_frontier_path(self, current_level: int) -> list[int] | None:
        """BFS from root to find shortest path to node with untried productive actions."""
        if self.root_hash is None:
            return None
        
        queue: deque[tuple[str, list[int]]] = deque([(self.root_hash, [])])
        visited = {self.root_hash}
        best: list[int] | None = None
        
        while queue:
            node_hash, path = queue.popleft()
            node = self.graph.get(node_hash)
            if not node:
                continue
            if node.levels_completed != current_level:
                continue
            
            productive_untried = [
                a for a in node.available_actions
                if a not in node.tried_actions and a not in node.noop_actions
            ]
            if productive_untried:
                if best is None or len(path) < len(best):
                    best = path
                continue
            
            for aid, next_hash in node.edges.items():
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((next_hash, path + [aid]))
        
        return best

    def _pick_action(
        self, untried: list[int], node: GraphNode, frame: FrameDataRaw
    ) -> int:
        """Pick next action. Uses heuristics based on action diversity and depth."""
        if len(untried) == 1:
            return untried[0]
        
        recent = self.current_path[-5:] if self.current_path else []
        
        if recent:
            last_action = recent[-1]
            run_length = 0
            for a in reversed(recent):
                if a == last_action:
                    run_length += 1
                else:
                    break
            
            if run_length < 3 and last_action in untried:
                return last_action
        
        for a in [3, 4, 1, 2]:
            if a in untried:
                return a
        return untried[0]

    def _build_result(
        self, win: bool, levels: int, level_actions: list[int]
    ) -> dict[str, Any]:
        rhae_scores = []
        if self.metadata:
            for i, actions in enumerate(level_actions):
                baseline = self.metadata.baseline_for_level(i)
                if baseline > 0 and actions > 0:
                    score = min((baseline / actions) ** 2 * 100, 115.0)
                else:
                    score = 0.0
                rhae_scores.append({
                    "level": i, "actions": actions, "baseline": baseline,
                    "score": score, "max_actions": self._budget(i),
                    "within_budget": actions <= self._budget(i),
                })

        total_rhae = sum(s["score"] for s in rhae_scores)
        max_rhae = 115.0 * (len(self.metadata.baseline_actions) if self.metadata else levels)

        return {
            "game_id": self.game_id,
            "win": win,
            "levels_completed": levels,
            "total_env_steps": self.total_steps,
            "level_actions": level_actions,
            "states_explored": len(self.graph),
            "solutions_found": len(self.level_solutions),
            "rhae": {
                "level_scores": rhae_scores,
                "total_score": total_rhae,
                "max_possible": max_rhae,
                "levels_completed": levels,
            },
        }
