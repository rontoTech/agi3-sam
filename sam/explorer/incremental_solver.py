"""Incremental solver: accumulates state graph across multiple episodes.

Key insight: The environment is DETERMINISTIC. We can run multiple episodes,
each time exploring different areas, and accumulate transitions in a persistent
graph. Once we've explored enough, we can find solutions via graph BFS.

This handles games with 3-strike GAME_OVER mechanics by:
1. Tracking which (state, action) transitions lead toward GAME_OVER
2. Marking "dangerous" states to avoid
3. Finding safe paths through the accumulated graph
4. Running multiple episodes to build complete coverage
"""

from __future__ import annotations

import hashlib
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
from sam.explorer import fast_frame_hash

logger = logging.getLogger(__name__)


@dataclass
class PersistentNode:
    """Node in the persistent state graph."""
    hash: str
    levels_completed: int
    available_actions: list[int]
    edges: dict[int, str] = field(default_factory=dict)
    noop_actions: set[int] = field(default_factory=set)
    is_game_over: bool = False
    strike_count: int = 0  # How many times visiting this state led toward GAME_OVER


class IncrementalSolver:
    """Accumulates state graph across episodes to find solutions.
    
    The solver runs multiple "exploration episodes" on the same deterministic
    environment. Each episode explores different areas, and the graph grows.
    
    Once a complete path from start to level-up is found in the graph,
    the solution is extracted and cached.
    """

    def __init__(
        self,
        game_id: str,
        environments_dir: str = "environment_files",
        max_episodes: int = 50,
        solutions_cache: Path | None = None,
    ) -> None:
        self.game_id = game_id
        self.environments_dir = environments_dir
        self.max_episodes = max_episodes
        self.metadata = find_metadata(game_id, Path(environments_dir))
        self.solutions_cache = solutions_cache
        
        self.graphs: dict[int, dict[str, PersistentNode]] = {}
        self.level_roots: dict[int, str] = {}
        self.level_solutions: dict[int, list[int]] = {}
        self.dangerous_transitions: set[tuple[str, int]] = set()
        self.game_over_paths: list[list[tuple[str, int]]] = []
        
        if solutions_cache and solutions_cache.exists():
            self._load_cache(solutions_cache)

    def _load_cache(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for k, v in data.get("level_plans", {}).items():
                self.level_solutions[int(k)] = list(v)
        except Exception:
            pass

    def _save_cache(self) -> None:
        if not self.solutions_cache:
            return
        self.solutions_cache.parent.mkdir(parents=True, exist_ok=True)
        full = []
        for i in sorted(self.level_solutions.keys()):
            full.extend(self.level_solutions[i])
        self.solutions_cache.write_text(json.dumps({
            "game_id": self.game_id,
            "level_plans": {str(k): v for k, v in sorted(self.level_solutions.items())},
            "full_path": full,
        }, indent=2))

    def _get_graph(self, level: int) -> dict[str, PersistentNode]:
        if level not in self.graphs:
            self.graphs[level] = {}
        return self.graphs[level]

    def _get_node(self, graph: dict[str, PersistentNode], frame: FrameDataRaw, h: str) -> PersistentNode:
        if h not in graph:
            graph[h] = PersistentNode(
                hash=h,
                levels_completed=frame.levels_completed,
                available_actions=[a for a in frame.available_actions if a != 0],
            )
        return graph[h]

    def _step(self, env: EnvironmentWrapper, action_id: int) -> FrameDataRaw | None:
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = self.game_id
        elif action.is_complex():
            action.set_data({"game_id": self.game_id, "x": 32, "y": 32})
        return env.step(action)

    def _replay_prefix(self, env: EnvironmentWrapper) -> tuple[FrameDataRaw | None, int]:
        """Replay solutions for all previously solved levels."""
        frame = env.reset()
        if frame is None:
            return None, 0
        steps = 0
        for level in sorted(self.level_solutions.keys()):
            if frame.levels_completed != level:
                break
            for aid in self.level_solutions[level]:
                frame = self._step(env, aid)
                if frame is None:
                    return None, steps
                steps += 1
                if frame.state == GameState.GAME_OVER:
                    return frame, steps
        return frame, steps

    def solve_all(self) -> dict[str, Any]:
        """Solve all levels of the game."""
        num_levels = len(self.metadata.baseline_actions) if self.metadata else 7
        level_action_counts: list[int] = []
        
        for target_level in range(num_levels):
            if target_level in self.level_solutions:
                level_action_counts.append(len(self.level_solutions[target_level]))
                continue
            
            solved = self._solve_level(target_level)
            if not solved:
                logger.warning("Could not solve level %d", target_level)
                break
            level_action_counts.append(len(self.level_solutions[target_level]))
        
        self._save_cache()
        return self._build_result(level_action_counts)

    def _solve_level(self, target_level: int) -> bool:
        """Solve a single level using incremental exploration."""
        graph = self._get_graph(target_level)
        
        for episode in range(self.max_episodes):
            arcade = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=self.environments_dir,
            )
            env = arcade.make(self.game_id, save_recording=False)
            if env is None:
                return False
            
            frame, prefix_steps = self._replay_prefix(env)
            if frame is None or frame.state == GameState.GAME_OVER:
                continue
            if frame.levels_completed != target_level:
                continue
            
            start_hash = fast_frame_hash(frame)
            self.level_roots[target_level] = start_hash
            self._get_node(graph, frame, start_hash)
            
            self._explore_episode(env, frame, graph, target_level, episode)
            
            solution = self._find_solution_in_graph(graph, target_level)
            if solution is not None:
                self.level_solutions[target_level] = solution
                logger.info(
                    "Level %d solved! Episodes: %d, Path length: %d, Graph size: %d",
                    target_level, episode + 1, len(solution), len(graph)
                )
                return True
            
            logger.info(
                "Level %d episode %d: graph has %d states, no solution yet",
                target_level, episode + 1, len(graph)
            )
        
        return False

    def _explore_episode(
        self, env: EnvironmentWrapper, frame: FrameDataRaw,
        graph: dict[str, PersistentNode], level: int, episode: int
    ) -> None:
        """Run one exploration episode, adding to the persistent graph."""
        path: list[tuple[str, int]] = []
        budget = self._budget(level) if self.metadata else 1000
        steps = 0
        
        strategies = [
            self._strategy_dfs,
            self._strategy_frontier_seek,
            self._strategy_random_walk,
        ]
        strategy = strategies[episode % len(strategies)]
        
        while steps < budget:
            if frame.state == GameState.WIN:
                break
            if frame.state == GameState.GAME_OVER:
                self._mark_dangerous_path(path)
                break
            if frame.levels_completed != level:
                break
            
            cur_hash = fast_frame_hash(frame)
            node = self._get_node(graph, frame, cur_hash)
            
            aid = strategy(node, graph, path, cur_hash, episode)
            if aid is None:
                break
            
            node.edges.setdefault(aid, "")
            path.append((cur_hash, aid))
            
            frame = self._step(env, aid)
            if frame is None:
                break
            steps += 1
            
            new_hash = fast_frame_hash(frame)
            if new_hash == cur_hash:
                node.noop_actions.add(aid)
                path.pop()
            else:
                node.edges[aid] = new_hash
                self._get_node(graph, frame, new_hash)
                
                if frame.levels_completed > level:
                    break

    def _strategy_dfs(
        self, node: PersistentNode, graph: dict[str, PersistentNode],
        path: list[tuple[str, int]], cur_hash: str, episode: int
    ) -> int | None:
        """DFS: try untried safe actions, prefer momentum."""
        untried = self._safe_untried(node, cur_hash)
        if not untried:
            return self._backtrack_or_redirect(node, graph, path, episode)
        
        if path:
            last_action = path[-1][1]
            if last_action in untried:
                return last_action
        
        return untried[0]

    def _strategy_frontier_seek(
        self, node: PersistentNode, graph: dict[str, PersistentNode],
        path: list[tuple[str, int]], cur_hash: str, episode: int
    ) -> int | None:
        """Seek nodes with untried actions via known edges."""
        untried = self._safe_untried(node, cur_hash)
        if untried:
            return untried[episode % len(untried)]
        
        for aid, next_hash in node.edges.items():
            if not next_hash or (cur_hash, aid) in self.dangerous_transitions:
                continue
            next_node = graph.get(next_hash)
            if next_node and self._safe_untried(next_node, next_hash):
                return aid
        
        return self._backtrack_or_redirect(node, graph, path, episode)

    def _strategy_random_walk(
        self, node: PersistentNode, graph: dict[str, PersistentNode],
        path: list[tuple[str, int]], cur_hash: str, episode: int
    ) -> int | None:
        """Random walk biased toward unexplored areas."""
        import random
        untried = self._safe_untried(node, cur_hash)
        if untried:
            return random.choice(untried)
        
        safe_actions = [
            a for a in node.available_actions
            if a not in node.noop_actions and (cur_hash, a) not in self.dangerous_transitions
        ]
        if safe_actions:
            return random.choice(safe_actions)
        return None

    def _safe_untried(self, node: PersistentNode, cur_hash: str) -> list[int]:
        """Get untried actions that are not known to be noops or dangerous."""
        tried = set(node.edges.keys()) | node.noop_actions
        return [
            a for a in node.available_actions
            if a not in tried and (cur_hash, a) not in self.dangerous_transitions
        ]

    def _backtrack_or_redirect(
        self, node: PersistentNode, graph: dict[str, PersistentNode],
        path: list[tuple[str, int]], episode: int
    ) -> int | None:
        """When stuck, try to move toward unexplored areas."""
        safe_edges = [
            (a, h) for a, h in node.edges.items()
            if h and (node.hash, a) not in self.dangerous_transitions
        ]
        if safe_edges:
            for a, h in safe_edges:
                n = graph.get(h)
                if n and self._safe_untried(n, h):
                    return a
            return safe_edges[episode % len(safe_edges)][0]
        return None

    def _mark_dangerous_path(self, path: list[tuple[str, int]]) -> None:
        """After GAME_OVER, mark transitions in the path as potentially dangerous."""
        self.game_over_paths.append(path[:])
        if len(path) >= 3:
            for state_hash, action in path[-3:]:
                self.dangerous_transitions.add((state_hash, action))
        elif path:
            state_hash, action = path[-1]
            self.dangerous_transitions.add((state_hash, action))

    def _find_solution_in_graph(
        self, graph: dict[str, PersistentNode], level: int
    ) -> list[int] | None:
        """BFS over accumulated graph to find path from root to level-up."""
        root = self.level_roots.get(level)
        if not root:
            return None
        
        queue: deque[tuple[str, list[int]]] = deque([(root, [])])
        visited = {root}
        
        while queue:
            node_hash, path = queue.popleft()
            node = graph.get(node_hash)
            if not node:
                continue
            
            if node.levels_completed > level:
                return path
            
            if len(path) > 500:
                continue
            
            for aid, next_hash in node.edges.items():
                if not next_hash or next_hash in visited:
                    continue
                if (node_hash, aid) in self.dangerous_transitions:
                    continue
                next_node = graph.get(next_hash)
                if next_node and not next_node.is_game_over:
                    visited.add(next_hash)
                    queue.append((next_hash, path + [aid]))
        
        return None

    def _budget(self, level: int) -> int:
        if self.metadata:
            return self.metadata.max_actions_for_level(level)
        return 1000

    def _build_result(self, level_actions: list[int]) -> dict[str, Any]:
        levels = len(level_actions)
        win = self.metadata is not None and levels >= len(self.metadata.baseline_actions)
        
        rhae_scores = []
        if self.metadata:
            for i, actions in enumerate(level_actions):
                baseline = self.metadata.baseline_for_level(i)
                score = min((baseline / actions) ** 2 * 100, 115.0) if actions > 0 else 0
                rhae_scores.append({
                    "level": i, "actions": actions, "baseline": baseline,
                    "score": score, "within_budget": actions <= self._budget(i),
                })

        total_rhae = sum(s["score"] for s in rhae_scores)
        max_rhae = 115.0 * (len(self.metadata.baseline_actions) if self.metadata else levels)
        
        return {
            "game_id": self.game_id,
            "win": win,
            "levels_completed": levels,
            "level_actions": level_actions,
            "solutions": {str(k): len(v) for k, v in self.level_solutions.items()},
            "graph_sizes": {str(k): len(v) for k, v in self.graphs.items()},
            "dangerous_transitions": len(self.dangerous_transitions),
            "rhae": {
                "level_scores": rhae_scores,
                "total_score": total_rhae,
                "max_possible": max_rhae,
            },
        }
