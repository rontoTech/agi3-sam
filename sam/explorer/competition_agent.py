"""Competition agent: general-purpose ARC-AGI-3 solver.

This agent combines:
1. Systematic graph exploration (state-action BFS)
2. Known-path replay for solved levels
3. Iterative deepening with budget awareness
4. Visual change detection for action prioritization
5. Multi-game support (keyboard + click actions)
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


@dataclass
class StateInfo:
    """Compact state tracking for exploration."""
    hash: str
    levels_completed: int
    available_actions: list[int]
    tried: set[int] = field(default_factory=set)
    edges: dict[int, str] = field(default_factory=dict)
    visual_changes: dict[int, int] = field(default_factory=dict)
    parent_hash: str | None = None
    parent_action: int | None = None
    depth: int = 0


@dataclass
class LevelSolution:
    """Stores a validated solution path for a level."""
    level: int
    actions: list[int]
    action_count: int
    discovered_at_step: int


class CompetitionAgent:
    """General-purpose ARC-AGI-3 competition agent.
    
    The agent uses iterative deepening DFS with BFS path optimization:
    1. DFS to discover level-up transitions quickly
    2. Once a path is found, BFS from level start to find the shortest path
    3. Replay optimized paths for maximum RHAE
    
    No neural network needed - pure systematic exploration.
    """

    def __init__(
        self,
        game_id: str,
        environments_dir: str | Path = "environment_files",
        max_depth: int = 200,
        solutions_cache: Path | None = None,
    ) -> None:
        self.game_id = game_id
        self.environments_dir = str(environments_dir)
        self.max_depth = max_depth
        self.solutions_cache = solutions_cache

        self.metadata = find_metadata(game_id, Path(self.environments_dir))
        
        self.states: dict[str, StateInfo] = {}
        self.level_solutions: dict[int, LevelSolution] = {}
        self.total_env_steps = 0
        
        if solutions_cache and solutions_cache.exists():
            self._load_solutions(solutions_cache)

    def _load_solutions(self, path: Path) -> None:
        """Load pre-computed solutions from cache."""
        try:
            data = json.loads(path.read_text())
            for level_str, actions in data.get("level_plans", {}).items():
                level = int(level_str)
                self.level_solutions[level] = LevelSolution(
                    level=level,
                    actions=actions,
                    action_count=len(actions),
                    discovered_at_step=0,
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load solutions cache: %s", e)

    def _save_solutions(self, path: Path) -> None:
        """Save discovered solutions to cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        plans = {}
        full_path = []
        for level in sorted(self.level_solutions.keys()):
            plans[str(level)] = self.level_solutions[level].actions
            full_path.extend(self.level_solutions[level].actions)
        path.write_text(json.dumps({
            "game_id": self.game_id,
            "level_plans": plans,
            "full_path": full_path,
        }, indent=2))

    def solve(self) -> dict[str, Any]:
        """Main solve loop: explore and optimize until WIN or budget exhausted."""
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

        level_actions_list: list[int] = []
        current_level = 0
        current_level_steps = 0
        
        while frame.state != GameState.WIN:
            level = frame.levels_completed
            max_budget = self._budget_for_level(level)

            if level in self.level_solutions:
                frame, steps = self._replay_solution(env, frame, level)
                if frame is None:
                    break
                if frame.levels_completed > level:
                    level_actions_list.append(steps)
                    current_level_steps = 0
                    logger.info("Level %d replayed in %d steps", level, steps)
                    continue
                else:
                    logger.warning("Replay failed for level %d, exploring", level)
                    del self.level_solutions[level]

            frame, steps, solved = self._explore_level(env, frame, max_budget)
            if frame is None:
                break
            
            if solved:
                level_actions_list.append(steps)
                current_level_steps = 0
            else:
                logger.warning("Failed to solve level %d", level)
                break

        win = frame is not None and frame.state == GameState.WIN
        levels = frame.levels_completed if frame else 0
        
        if win and current_level_steps > 0:
            level_actions_list.append(current_level_steps)

        if self.solutions_cache:
            self._save_solutions(self.solutions_cache)

        result = self._build_result(win, levels, level_actions_list)
        return result

    def _budget_for_level(self, level: int) -> int:
        if self.metadata:
            return self.metadata.max_actions_for_level(level)
        return self.max_depth * 5

    def _replay_solution(
        self, env: EnvironmentWrapper, frame: FrameDataRaw, level: int
    ) -> tuple[FrameDataRaw | None, int]:
        """Replay a known solution path."""
        solution = self.level_solutions[level]
        steps = 0
        for action_id in solution.actions:
            action = GameAction.from_id(action_id)
            if action.is_simple():
                action.action_data.game_id = self.game_id
            elif action.is_complex():
                action.set_data({"game_id": self.game_id, "x": 32, "y": 32})
            
            frame = env.step(action)
            if frame is None:
                return None, steps
            steps += 1
            self.total_env_steps += 1

            if frame.state == GameState.GAME_OVER:
                action = GameAction.from_id(GameAction.RESET.value)
                action.action_data.game_id = self.game_id
                frame = env.step(action)
                return frame, steps + 1

            if frame.levels_completed > level:
                return frame, steps

        return frame, steps

    def _explore_level(
        self, env: EnvironmentWrapper, frame: FrameDataRaw, budget: int
    ) -> tuple[FrameDataRaw | None, int, bool]:
        """Explore current level using iterative deepening DFS with BFS optimization."""
        level = frame.levels_completed
        start_hash = fast_frame_hash(frame)
        
        self.states = {}
        self._get_state(frame, start_hash, depth=0)
        
        steps = 0
        current_path: list[int] = []
        best_solution: list[int] | None = None
        
        while steps < budget:
            cur_hash = fast_frame_hash(frame)
            state = self._get_state(frame, cur_hash, depth=len(current_path))

            action_id = self._pick_exploration_action(state, current_path, frame)
            
            if action_id == GameAction.RESET.value:
                action = GameAction.from_id(action_id)
                action.action_data.game_id = self.game_id
                frame = env.step(action)
                if frame is None:
                    return None, steps, False
                steps += 1
                self.total_env_steps += 1
                current_path = []
                continue

            action = GameAction.from_id(action_id)
            if action.is_simple():
                action.action_data.game_id = self.game_id
            elif action.is_complex():
                action.set_data({"game_id": self.game_id, "x": 32, "y": 32})

            prev_frame = frame
            frame = env.step(action)
            if frame is None:
                return None, steps, False
            steps += 1
            self.total_env_steps += 1
            current_path.append(action_id)

            new_hash = fast_frame_hash(frame)
            state.tried.add(action_id)
            state.edges[action_id] = new_hash
            state.visual_changes[action_id] = frame_visual_diff(prev_frame, frame)
            
            new_state = self._get_state(frame, new_hash, depth=len(current_path))
            new_state.parent_hash = cur_hash
            new_state.parent_action = action_id

            if frame.state == GameState.GAME_OVER:
                action = GameAction.from_id(GameAction.RESET.value)
                action.action_data.game_id = self.game_id
                frame = env.step(action)
                if frame is None:
                    return None, steps, False
                steps += 1
                self.total_env_steps += 1
                current_path = []
                continue

            if frame.levels_completed > level:
                if best_solution is None or len(current_path) < len(best_solution):
                    best_solution = current_path[:]
                    self.level_solutions[level] = LevelSolution(
                        level=level,
                        actions=best_solution,
                        action_count=len(best_solution),
                        discovered_at_step=self.total_env_steps,
                    )
                    logger.info(
                        "Level %d solved! Path length: %d, total steps used: %d",
                        level, len(best_solution), steps
                    )
                return frame, steps, True

            if len(current_path) >= min(self.max_depth, budget - steps):
                action = GameAction.from_id(GameAction.RESET.value)
                action.action_data.game_id = self.game_id
                frame = env.step(action)
                if frame is None:
                    return None, steps, False
                steps += 1
                self.total_env_steps += 1
                current_path = []

        return frame, steps, False

    def _get_state(self, frame: FrameDataRaw, h: str, depth: int = 0) -> StateInfo:
        if h not in self.states:
            self.states[h] = StateInfo(
                hash=h,
                levels_completed=frame.levels_completed,
                available_actions=[a for a in frame.available_actions if a != 0],
                depth=depth,
            )
        return self.states[h]

    def _pick_exploration_action(
        self, state: StateInfo, current_path: list[int], frame: FrameDataRaw
    ) -> int:
        """Pick next action using DFS with visual-change priority."""
        untried = [a for a in state.available_actions if a not in state.tried]
        
        if untried:
            if state.visual_changes:
                scored = []
                for a in untried:
                    similar_score = 0
                    for other_state in self.states.values():
                        if a in other_state.visual_changes:
                            similar_score = max(similar_score, other_state.visual_changes[a])
                    scored.append((a, similar_score))
                scored.sort(key=lambda x: -x[1])
                return scored[0][0]
            return untried[0]

        if self._is_cycle(state.hash, current_path):
            return GameAction.RESET.value

        frontier = self._find_nearest_frontier(state.hash)
        if frontier is not None:
            return GameAction.RESET.value

        return GameAction.RESET.value

    def _is_cycle(self, cur_hash: str, path: list[int]) -> bool:
        """Detect cycles in current path."""
        if len(path) < 4:
            return False
        recent_hashes = []
        h = cur_hash
        for _ in range(min(10, len(path))):
            if h in recent_hashes:
                return True
            recent_hashes.append(h)
            state = self.states.get(h)
            if state and state.parent_hash:
                h = state.parent_hash
            else:
                break
        return False

    def _find_nearest_frontier(self, start: str) -> str | None:
        """BFS to find nearest state with untried actions."""
        queue: deque[str] = deque([start])
        visited = {start}
        while queue:
            h = queue.popleft()
            state = self.states.get(h)
            if not state:
                continue
            untried = [a for a in state.available_actions if a not in state.tried]
            if untried and h != start:
                return h
            for _, next_h in state.edges.items():
                if next_h not in visited:
                    visited.add(next_h)
                    queue.append(next_h)
        return None

    def _build_result(
        self, win: bool, levels: int, level_actions: list[int]
    ) -> dict[str, Any]:
        """Build result summary with RHAE computation."""
        rhae_scores = []
        if self.metadata:
            for i, actions in enumerate(level_actions):
                baseline = self.metadata.baseline_for_level(i)
                if baseline > 0 and actions > 0:
                    score = min((baseline / actions) ** 2 * 100, 115.0)
                else:
                    score = 0.0
                rhae_scores.append({
                    "level": i,
                    "actions": actions,
                    "baseline": baseline,
                    "score": score,
                    "max_actions": self._budget_for_level(i),
                    "within_budget": actions <= self._budget_for_level(i),
                })

        total_rhae = sum(s["score"] for s in rhae_scores)
        max_rhae = 115.0 * (len(self.metadata.baseline_actions) if self.metadata else levels)

        return {
            "game_id": self.game_id,
            "win": win,
            "levels_completed": levels,
            "total_env_steps": self.total_env_steps,
            "level_actions": level_actions,
            "states_explored": len(self.states),
            "rhae": {
                "level_scores": rhae_scores,
                "total_score": total_rhae,
                "max_possible": max_rhae,
                "levels_completed": levels,
            },
        }
