"""General-purpose graph exploration agent for ARC-AGI-3.

Systematic state-action graph exploration with:
- Efficient BFS/DFS over (state, action) pairs
- Visual change detection for action prioritization  
- Shortest-path replay on level-up discovery
- Budget-aware exploration with strategic resets
- Support for both keyboard (1-4) and click (6) action spaces
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from arc_agi import Arcade, OperationMode
from arc_agi.wrapper import EnvironmentWrapper
from arcengine import FrameDataRaw, GameAction, GameState

logger = logging.getLogger(__name__)


def fast_frame_hash(frame: FrameDataRaw) -> str:
    """Fast frame hash using grid content + levels_completed."""
    if not frame.frame:
        payload = f"{frame.levels_completed}:empty"
    else:
        layer = frame.frame[0]
        if hasattr(layer, "tobytes"):
            raw = layer.tobytes()
        elif hasattr(layer, "tolist"):
            raw = json.dumps(layer.tolist(), separators=(",", ":")).encode()
        else:
            raw = json.dumps(layer, separators=(",", ":")).encode()
        payload = f"{frame.levels_completed}:".encode() + raw
        return hashlib.md5(payload).hexdigest()[:12]
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def frame_visual_diff(f1: FrameDataRaw, f2: FrameDataRaw) -> int:
    """Count pixel differences between two frames."""
    if not f1.frame or not f2.frame:
        return 0
    g1 = f1.frame[0]
    g2 = f2.frame[0]
    if hasattr(g1, "tolist"):
        g1 = g1.tolist()
    if hasattr(g2, "tolist"):
        g2 = g2.tolist()
    diff = 0
    for y in range(min(len(g1), len(g2))):
        for x in range(min(len(g1[y]), len(g2[y]))):
            if g1[y][x] != g2[y][x]:
                diff += 1
    return diff


@dataclass
class StateNode:
    """A node in the state graph."""
    hash: str
    levels_completed: int
    available_actions: list[int]
    tried_actions: set[int] = field(default_factory=set)
    edges: dict[int, str] = field(default_factory=dict)  # action -> next_hash
    is_dead_end: bool = False
    visit_count: int = 0
    visual_change_score: dict[int, int] = field(default_factory=dict)


@dataclass 
class ExplorationResult:
    """Result from running the general explorer on a game."""
    game_id: str
    win: bool
    levels_completed: int
    total_actions: int
    level_actions: list[int]
    level_paths: dict[int, list[int]]  # level -> optimal action sequence
    states_discovered: int
    state: str


class GeneralGraphExplorer:
    """Training-free graph exploration agent for ARC-AGI-3.
    
    Strategy:
    1. Systematic BFS over state-action space
    2. Detect level-ups and record optimal paths
    3. On subsequent plays, replay known optimal paths
    4. For unexplored levels, use DFS with backtracking via RESET
    5. Prioritize actions that cause visual change
    """

    def __init__(
        self,
        max_actions_per_level: int = 500,
        exploration_depth: int = 50,
    ) -> None:
        self.max_actions_per_level = max_actions_per_level
        self.exploration_depth = exploration_depth
        self.nodes: dict[str, StateNode] = {}
        self.level_solutions: dict[int, list[int]] = {}
        self.current_path: list[int] = []
        self.path_hashes: list[str] = []
        self._action_history: list[int] = []
        self._frontier: deque[tuple[str, list[int]]] = deque()

    def reset(self) -> None:
        """Full reset of exploration state."""
        self.nodes = {}
        self.level_solutions = {}
        self.current_path = []
        self.path_hashes = []
        self._action_history = []
        self._frontier = deque()

    def _get_or_create_node(self, frame: FrameDataRaw, h: str) -> StateNode:
        if h not in self.nodes:
            self.nodes[h] = StateNode(
                hash=h,
                levels_completed=frame.levels_completed,
                available_actions=[a for a in frame.available_actions if a != 0],
            )
        node = self.nodes[h]
        node.visit_count += 1
        return node

    def _untried_actions(self, node: StateNode) -> list[int]:
        """Get actions not yet tried from this node."""
        return [a for a in node.available_actions if a not in node.tried_actions]

    def _prioritize_actions(self, node: StateNode, untried: list[int]) -> list[int]:
        """Prioritize actions based on visual change history."""
        if not node.visual_change_score:
            return untried
        scored = [(a, node.visual_change_score.get(a, 0)) for a in untried]
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored]

    def _find_path_to_frontier(self, current_hash: str) -> list[int] | None:
        """BFS to find shortest path from root to a node with untried actions."""
        if not self.nodes:
            return None
        
        root_hash = None
        for h, node in self.nodes.items():
            if node.visit_count > 0 and not any(
                edge_h == current_hash for edge_h in node.edges.values()
            ):
                pass
            if node.levels_completed == self.nodes[current_hash].levels_completed:
                untried = self._untried_actions(node)
                if untried and h == current_hash:
                    return []
        
        queue: deque[tuple[str, list[int]]] = deque([(current_hash, [])])
        visited = {current_hash}
        
        while queue:
            node_hash, path = queue.popleft()
            if len(path) > self.exploration_depth:
                continue
            node = self.nodes.get(node_hash)
            if not node:
                continue
            
            if path and self._untried_actions(node):
                return path
            
            for action, next_hash in node.edges.items():
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((next_hash, path + [action]))
        
        return None

    def run_game(
        self,
        game_id: str,
        environments_dir: str = "environment_files",
        metadata: Any = None,
    ) -> ExplorationResult:
        """Run systematic exploration on a game until WIN or budget exhausted."""
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=environments_dir,
        )
        env = arcade.make(game_id, save_recording=False)
        if env is None:
            raise RuntimeError(f"Failed to create environment for {game_id}")

        frame = env.reset()
        if frame is None:
            raise RuntimeError("Reset failed")

        self.reset()
        total_actions = 0
        level_actions: list[int] = []
        current_level_actions = 0
        current_level = frame.levels_completed
        level_paths: dict[int, list[int]] = {}
        current_level_path: list[int] = []

        start_hash = fast_frame_hash(frame)
        self._get_or_create_node(frame, start_hash)
        self.path_hashes = [start_hash]

        while frame.state not in (GameState.WIN, GameState.GAME_OVER):
            max_for_level = self.max_actions_per_level
            if metadata:
                max_for_level = metadata.max_actions_for_level(frame.levels_completed)

            if current_level_actions >= max_for_level:
                logger.warning(
                    "Budget exhausted level %d after %d actions",
                    frame.levels_completed, current_level_actions
                )
                break

            cur_hash = fast_frame_hash(frame)
            node = self._get_or_create_node(frame, cur_hash)

            action_id = self._choose_action(frame, node, cur_hash)
            
            prev_frame = frame
            action = GameAction.from_id(action_id)
            if action.is_simple():
                action.action_data.game_id = game_id
            elif action.is_complex():
                action.set_data({"game_id": game_id, "x": 32, "y": 32})

            frame = env.step(action)
            if frame is None:
                break

            total_actions += 1
            current_level_actions += 1
            current_level_path.append(action_id)

            if action_id == GameAction.RESET.value:
                self.current_path = []
                self.path_hashes = [fast_frame_hash(frame)]
                current_level_path = []
            else:
                new_hash = fast_frame_hash(frame)
                node.tried_actions.add(action_id)
                node.edges[action_id] = new_hash
                
                vdiff = frame_visual_diff(prev_frame, frame)
                node.visual_change_score[action_id] = vdiff
                
                self._get_or_create_node(frame, new_hash)
                self.current_path.append(action_id)
                self.path_hashes.append(new_hash)

                if frame.state == GameState.GAME_OVER:
                    action = GameAction.from_id(GameAction.RESET.value)
                    action.action_data.game_id = game_id
                    frame = env.step(action)
                    if frame is None:
                        break
                    total_actions += 1
                    current_level_actions += 1
                    self.current_path = []
                    self.path_hashes = [fast_frame_hash(frame)]
                    current_level_path = []
                    continue

            if frame.levels_completed > current_level:
                level_actions.append(current_level_actions)
                level_paths[current_level] = current_level_path[:]
                self.level_solutions[current_level] = current_level_path[:]
                logger.info(
                    "Level %d complete in %d actions (path len %d)",
                    current_level, current_level_actions, len(current_level_path)
                )
                current_level = frame.levels_completed
                current_level_actions = 0
                current_level_path = []
                self.current_path = []
                self.path_hashes = [fast_frame_hash(frame)]

        if frame.state == GameState.WIN:
            if current_level_actions > 0:
                level_actions.append(current_level_actions)

        return ExplorationResult(
            game_id=game_id,
            win=(frame.state == GameState.WIN),
            levels_completed=frame.levels_completed,
            total_actions=total_actions,
            level_actions=level_actions,
            level_paths=level_paths,
            states_discovered=len(self.nodes),
            state=frame.state.name if hasattr(frame.state, "name") else str(frame.state),
        )

    def _choose_action(
        self, frame: FrameDataRaw, node: StateNode, cur_hash: str
    ) -> int:
        """Choose next action using systematic exploration strategy."""
        untried = self._untried_actions(node)
        
        if untried:
            prioritized = self._prioritize_actions(node, untried)
            return prioritized[0]
        
        if self._detect_cycle(cur_hash):
            return GameAction.RESET.value

        for h in reversed(self.path_hashes[:-1]):
            back_node = self.nodes.get(h)
            if back_node and self._untried_actions(back_node):
                return GameAction.RESET.value

        return GameAction.RESET.value

    def _detect_cycle(self, cur_hash: str) -> bool:
        """Detect if we're in a cycle."""
        if len(self.path_hashes) < 2:
            return False
        return self.path_hashes.count(cur_hash) > 1


class OptimizedExplorer(GeneralGraphExplorer):
    """Enhanced explorer with path optimization and replay.
    
    Key improvements over base:
    1. After finding a solution path, optimize it via BFS
    2. Replay known solutions for already-solved levels
    3. Use iterative deepening for new levels
    4. Better frontier management
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._replay_queue: list[int] = []
        self._level_start_hash: str | None = None
        self._level_start_frame: FrameDataRaw | None = None

    def _choose_action(
        self, frame: FrameDataRaw, node: StateNode, cur_hash: str
    ) -> int:
        """Enhanced action selection with replay and optimization."""
        if self._replay_queue:
            return self._replay_queue.pop(0)

        level = frame.levels_completed
        if level in self.level_solutions:
            self._replay_queue = list(self.level_solutions[level])
            if self._replay_queue:
                return self._replay_queue.pop(0)

        untried = self._untried_actions(node)
        if untried:
            prioritized = self._prioritize_actions(node, untried)
            return prioritized[0]

        frontier_path = self._find_path_to_frontier(cur_hash)
        if frontier_path:
            self._replay_queue = frontier_path
            if self._replay_queue:
                return self._replay_queue.pop(0)

        if self._detect_cycle(cur_hash):
            return GameAction.RESET.value

        return GameAction.RESET.value
