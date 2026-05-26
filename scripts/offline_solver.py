#!/usr/bin/env python3
"""Offline multi-episode solver for ARC-AGI-3 games.

Runs many episodes to accumulate a complete state graph,
then finds shortest paths via BFS. Designed for offline computation
(unlimited time) to pre-compute solutions before competition.

Usage:
    python scripts/offline_solver.py --game tr87-cd924810 --episodes 500
    python scripts/offline_solver.py --all --episodes 200
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("solver")


def fhash(frame) -> str:
    layer = frame.frame[0]
    raw = layer.tobytes() if hasattr(layer, "tobytes") else json.dumps(layer.tolist()).encode()
    return hashlib.md5(f"{frame.levels_completed}:".encode() + raw).hexdigest()[:12]


class OfflineMultiEpisodeSolver:
    """Accumulates state graph across many episodes to find solutions."""

    def __init__(self, game_id: str, env_dir: str = "environment_files") -> None:
        self.game_id = game_id
        self.env_dir = env_dir
        self.graphs: dict[int, dict[str, dict[int, str]]] = {}
        self.level_roots: dict[int, str] = {}
        self.level_solutions: dict[int, list[int]] = {}
        self.level_up_states: dict[int, set[str]] = {}

    def solve_level(
        self,
        level: int,
        max_episodes: int = 500,
        steps_per_episode: int = 300,
        time_limit: float = 300.0,
    ) -> list[int] | None:
        """Solve a single level using multi-episode graph accumulation."""
        if level not in self.graphs:
            self.graphs[level] = {}
        graph = self.graphs[level]
        
        if level not in self.level_up_states:
            self.level_up_states[level] = set()
        
        deadline = time.monotonic() + time_limit
        
        for episode in range(max_episodes):
            if time.monotonic() > deadline:
                break
            
            self._run_exploration_episode(level, graph, steps_per_episode, episode)
            
            solution = self._find_solution(level)
            if solution is not None:
                self.level_solutions[level] = solution
                logger.info(
                    "Level %d solved! Episode %d, path=%d, graph=%d states",
                    level, episode + 1, len(solution), len(graph),
                )
                return solution
            
            if (episode + 1) % 50 == 0:
                logger.info(
                    "Level %d: episode %d, graph=%d states, no solution yet",
                    level, episode + 1, len(graph),
                )
        
        return None

    def _run_exploration_episode(
        self, level: int, graph: dict, max_steps: int, episode: int
    ) -> None:
        """Run one exploration episode, adding to persistent graph."""
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=self.env_dir,
        )
        env = arcade.make(self.game_id, save_recording=False)
        if env is None:
            return
        
        frame = env.reset()
        if frame is None:
            return
        
        for prev_level in range(level):
            if prev_level not in self.level_solutions:
                return
            for aid in self.level_solutions[prev_level]:
                action = GameAction.from_id(aid)
                if action.is_simple():
                    action.action_data.game_id = self.game_id
                elif action.is_complex():
                    action.set_data({"game_id": self.game_id, "x": 32, "y": 32})
                frame = env.step(action)
                if frame is None or frame.state == GameState.GAME_OVER:
                    return
        
        if frame.levels_completed != level:
            return
        
        root_hash = fhash(frame)
        self.level_roots[level] = root_hash
        if root_hash not in graph:
            graph[root_hash] = {}
        
        path = []
        steps = 0
        avail = [a for a in frame.available_actions if a != 0]
        
        strategies = [
            self._dfs_momentum,
            self._random_walk,
            self._breadth_priority,
            self._reverse_momentum,
            self._alternating,
        ]
        strategy = strategies[episode % len(strategies)]
        
        while steps < max_steps:
            if frame.state == GameState.GAME_OVER:
                break
            if frame.levels_completed > level:
                cur_hash = fhash(frame)
                self.level_up_states[level].add(cur_hash)
                break
            
            cur_hash = fhash(frame)
            if cur_hash not in graph:
                graph[cur_hash] = {}
            
            node = graph[cur_hash]
            untried = [a for a in avail if a not in node]
            
            aid = strategy(untried, node, avail, path, episode, steps)
            
            before_hash = cur_hash
            action = GameAction.from_id(aid)
            if action.is_simple():
                action.action_data.game_id = self.game_id
            elif action.is_complex():
                x, y = random.randint(0, 63), random.randint(0, 63)
                action.set_data({"game_id": self.game_id, "x": x, "y": y})
            
            frame = env.step(action)
            if frame is None:
                break
            steps += 1
            
            new_hash = fhash(frame)
            if new_hash != before_hash:
                node[aid] = new_hash
                path.append(aid)
            elif aid not in node:
                node[aid] = before_hash

    def _dfs_momentum(self, untried, node, avail, path, episode, steps):
        if untried:
            if path and path[-1] in untried:
                return path[-1]
            return untried[0]
        if path and path[-1] in avail:
            return path[-1]
        return random.choice(avail) if avail else 1

    def _random_walk(self, untried, node, avail, path, episode, steps):
        if untried and random.random() < 0.7:
            return random.choice(untried)
        return random.choice(avail) if avail else 1

    def _breadth_priority(self, untried, node, avail, path, episode, steps):
        if untried:
            return untried[steps % len(untried)]
        return avail[steps % len(avail)] if avail else 1

    def _reverse_momentum(self, untried, node, avail, path, episode, steps):
        OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}
        if untried:
            if path:
                opp = OPPOSITE.get(path[-1])
                if opp in untried:
                    return opp
            return untried[0]
        return random.choice(avail) if avail else 1

    def _alternating(self, untried, node, avail, path, episode, steps):
        if untried:
            return untried[episode % len(untried)]
        pattern = [1, 3, 2, 4, 1, 4, 2, 3]
        return pattern[steps % len(pattern)] if avail else 1

    def _find_solution(self, level: int) -> list[int] | None:
        """BFS on accumulated graph to find path from root to level-up."""
        root = self.level_roots.get(level)
        if not root:
            return None
        
        level_ups = self.level_up_states.get(level, set())
        if not level_ups:
            return None
        
        graph = self.graphs[level]
        
        queue: deque[tuple[str, list[int]]] = deque([(root, [])])
        visited = {root}
        
        while queue:
            h, path = queue.popleft()
            if len(path) > 500:
                continue
            
            node = graph.get(h, {})
            for aid, next_h in node.items():
                if next_h in level_ups:
                    return path + [aid]
                if next_h not in visited and next_h in graph:
                    visited.add(next_h)
                    queue.append((next_h, path + [aid]))
        
        return None

    def solve_all(
        self,
        max_episodes_per_level: int = 500,
        steps_per_episode: int = 300,
        time_per_level: float = 300.0,
    ) -> dict[str, list[int]]:
        """Solve all levels sequentially."""
        from sam.config import find_metadata
        meta = find_metadata(self.game_id, Path(self.env_dir))
        if not meta:
            return {}
        
        num_levels = len(meta.baseline_actions)
        
        for level in range(num_levels):
            if level in self.level_solutions:
                continue
            
            solution = self.solve_level(
                level, max_episodes_per_level, steps_per_episode, time_per_level
            )
            if solution is None:
                logger.warning("Could not solve level %d", level)
                break
        
        return {str(k): v for k, v in self.level_solutions.items()}


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Offline Multi-Episode Solver")
    parser.add_argument("--game", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--output-dir", default="sam/checkpoints/solutions")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        games = []
        for meta_path in sorted(Path("environment_files").rglob("metadata.json")):
            data = json.loads(meta_path.read_text())
            if "game_id" in data:
                games.append(data["game_id"])
    elif args.game:
        games = [args.game]
    else:
        parser.error("Specify --game or --all")
        return
    
    for game_id in games:
        logger.info("=" * 50)
        logger.info("Solving: %s", game_id)
        
        solver = OfflineMultiEpisodeSolver(game_id)
        solutions = solver.solve_all(
            max_episodes_per_level=args.episodes,
            steps_per_episode=args.steps,
            time_per_level=args.time_limit,
        )
        
        if solutions:
            cache_path = output_dir / f"{game_id.replace('-', '_')}_solutions.json"
            full_path = []
            for i in sorted(int(k) for k in solutions.keys()):
                full_path.extend(solutions[str(i)])
            cache_path.write_text(json.dumps({
                "game_id": game_id,
                "level_plans": solutions,
                "full_path": full_path,
            }, indent=2))
            logger.info("Saved %d level solutions to %s", len(solutions), cache_path)
        else:
            logger.warning("No solutions found for %s", game_id)


if __name__ == "__main__":
    main()
