"""MCTS-style planner using the learned world model.

This is the core intelligence of SAM - it simulates action sequences
internally using the world model, evaluates them with the value head,
and selects the best action without consuming environment steps.

Key features:
1. Monte Carlo Tree Search with world model rollouts
2. Progressive widening (explore promising branches deeper)
3. UCB1 selection with confidence-weighted exploration
4. Test-time training: update WM with each real transition
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from arcengine import FrameDataRaw, GameAction, GameState

from sam.core.sam_core import SamCore
from sam.utils import frame_hash, frame_to_model_input


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    z: torch.Tensor
    action_from_parent: int | None = None
    parent: MCTSNode | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    confidence: float = 0.5
    is_terminal: bool = False
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class WorldModelPlanner:
    """MCTS planner using learned world model for simulation.
    
    The planner:
    1. Encodes the current frame into latent space
    2. Uses MCTS to explore action sequences via world model
    3. Evaluates leaf nodes with the value head
    4. Returns the best action based on visit counts
    
    Key innovation: combines model-based planning with test-time learning.
    After each real transition, the world model is updated to reduce prediction
    error, making future planning more accurate.
    """

    def __init__(
        self,
        core: SamCore,
        num_simulations: int = 64,
        max_depth: int = 15,
        c_puct: float = 1.4,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.core = core
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = torch.device(device)
        
        self._prev_z: torch.Tensor | None = None
        self._prev_action: int | None = None
        self._online_optimizer: torch.optim.Optimizer | None = None
        self._transitions_seen = 0

    def select_action(
        self,
        frame: FrameDataRaw,
        available_actions: list[int],
        planning_budget_ms: float = 50.0,
    ) -> tuple[int, dict[str, Any]]:
        """Select best action using MCTS with world model rollouts."""
        self.core.eval()
        
        with torch.no_grad():
            z = self.core.encode(frame_to_model_input(frame))
        
        actions = [a for a in available_actions if a != 0]
        if not actions:
            return 0, {"reason": "no_actions"}
        
        if len(actions) == 1:
            return actions[0], {"reason": "single_action"}
        
        root = MCTSNode(z=z)
        deadline = time.monotonic() + planning_budget_ms / 1000.0
        
        sims_done = 0
        while sims_done < self.num_simulations and time.monotonic() < deadline:
            self._simulate(root, actions, depth=0)
            sims_done += 1
        
        best_action = self._select_final_action(root, actions)
        
        info = {
            "simulations": sims_done,
            "action_values": {
                a: root.children[a].value if a in root.children else 0.0
                for a in actions
            },
            "action_visits": {
                a: root.children[a].visit_count if a in root.children else 0
                for a in actions
            },
        }
        
        self._prev_z = z
        self._prev_action = best_action
        
        return best_action, info

    def update_on_transition(
        self,
        frame_before: FrameDataRaw,
        action_id: int,
        frame_after: FrameDataRaw,
        lr: float = 1e-3,
    ) -> float:
        """Online world model update after observing a real transition.
        
        This is test-time training: we update the WM to better predict
        this specific environment's dynamics.
        """
        self.core.train()
        
        z_before = self.core.encode(frame_to_model_input(frame_before))
        z_after = self.core.encode(frame_to_model_input(frame_after))
        
        z_pred, conf = self.core.world_model(z_before.squeeze(0), action_id)
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)
        z_target = z_after.detach()
        if z_target.dim() == 1:
            z_target = z_target.unsqueeze(0)
        
        wm_loss = F.mse_loss(z_pred, z_target)
        
        if self._online_optimizer is None:
            self._online_optimizer = torch.optim.Adam(
                list(self.core.world_model.parameters()) +
                list(self.core.value.parameters()),
                lr=lr,
            )
        
        self._online_optimizer.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.core.parameters(), 0.5)
        self._online_optimizer.step()
        
        self.core.eval()
        self._transitions_seen += 1
        return wm_loss.item()

    def _simulate(self, node: MCTSNode, actions: list[int], depth: int) -> float:
        """Run one MCTS simulation from node."""
        if depth >= self.max_depth or node.is_terminal:
            return self._evaluate_leaf(node)
        
        action = self._select_child_action(node, actions)
        
        if action not in node.children:
            child = self._expand(node, action)
            node.children[action] = child
            value = self._evaluate_leaf(child)
        else:
            child = node.children[action]
            value = self._simulate(child, actions, depth + 1)
        
        node.visit_count += 1
        node.value_sum += value
        return value

    def _select_child_action(self, node: MCTSNode, actions: list[int]) -> int:
        """UCB1 selection with confidence weighting."""
        unexplored = [a for a in actions if a not in node.children]
        if unexplored:
            return unexplored[0]
        
        best_score = float("-inf")
        best_action = actions[0]
        
        for action in actions:
            child = node.children.get(action)
            if child is None:
                return action
            
            exploit = child.value
            explore = self.c_puct * math.sqrt(
                math.log(node.visit_count + 1) / (child.visit_count + 1)
            )
            confidence_bonus = 0.1 * child.confidence
            
            score = exploit + explore + confidence_bonus
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    @torch.no_grad()
    def _expand(self, parent: MCTSNode, action: int) -> MCTSNode:
        """Expand by predicting next state with world model."""
        z_next, conf = self.core.world_model(parent.z.squeeze(0), action)
        z_next = self.core.skills.apply(z_next)
        
        return MCTSNode(
            z=z_next.unsqueeze(0) if z_next.dim() == 1 else z_next,
            action_from_parent=action,
            parent=parent,
            confidence=float(conf.item()) if hasattr(conf, "item") else float(conf),
        )

    @torch.no_grad()
    def _evaluate_leaf(self, node: MCTSNode) -> float:
        """Evaluate leaf node using value head."""
        z = node.z
        if z.dim() == 1:
            z = z.unsqueeze(0)
        scalar_v, vector_v = self.core.value(z)
        
        base_value = float(scalar_v.item())
        confidence_weight = node.confidence
        
        return base_value * confidence_weight

    def _select_final_action(self, root: MCTSNode, actions: list[int]) -> int:
        """Select final action based on visit counts (robust selection)."""
        if self.temperature <= 0:
            best_visits = -1
            best_action = actions[0]
            for a in actions:
                if a in root.children and root.children[a].visit_count > best_visits:
                    best_visits = root.children[a].visit_count
                    best_action = a
            return best_action
        
        visits = []
        for a in actions:
            v = root.children[a].visit_count if a in root.children else 0
            visits.append(v)
        
        if sum(visits) == 0:
            return actions[0]
        
        visits_t = torch.tensor(visits, dtype=torch.float32)
        probs = F.softmax(visits_t / self.temperature, dim=0)
        idx = int(torch.multinomial(probs, 1).item())
        return actions[idx]


class AdaptivePlanner:
    """Combines MCTS planning with graph-based exploration and test-time training.
    
    Strategy per level:
    1. First N steps: explore and build graph + update WM online
    2. When WM confidence is high: switch to MCTS planning
    3. If plan fails: fall back to exploration
    4. Track progress: if stuck, increase exploration
    """

    def __init__(
        self,
        core: SamCore,
        planning_budget_ms: float = 50.0,
        num_simulations: int = 32,
        device: str = "cpu",
    ) -> None:
        self.core = core
        self.planner = WorldModelPlanner(
            core,
            num_simulations=num_simulations,
            max_depth=10,
            device=device,
        )
        self.planning_budget_ms = planning_budget_ms
        
        self._wm_errors: list[float] = []
        self._level_steps = 0
        self._explore_phase_steps = 20
        self._stuck_counter = 0
        self._prev_hash: str | None = None

    def reset_level(self) -> None:
        """Reset state for new level."""
        self._wm_errors = []
        self._level_steps = 0
        self._stuck_counter = 0
        self._prev_hash = None

    def choose_action(
        self,
        frame: FrameDataRaw,
        prev_frame: FrameDataRaw | None = None,
    ) -> tuple[int, str]:
        """Choose action using adaptive strategy."""
        self._level_steps += 1
        
        if prev_frame is not None:
            cur_hash = frame_hash(frame)
            if cur_hash == self._prev_hash:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            self._prev_hash = cur_hash
        
        actions = [a for a in frame.available_actions if a != 0]
        if not actions:
            return GameAction.RESET.value, "no_actions"
        
        wm_confident = self._is_wm_reliable()
        
        if self._level_steps <= self._explore_phase_steps or not wm_confident:
            return self._explore_action(frame, actions), "explore"
        
        if self._stuck_counter > 3:
            self._explore_phase_steps += 10
            self._stuck_counter = 0
            return self._explore_action(frame, actions), "unstuck"
        
        action_id, info = self.planner.select_action(
            frame, actions, self.planning_budget_ms
        )
        return action_id, "mcts"

    def on_transition(
        self, before: FrameDataRaw, action: int, after: FrameDataRaw
    ) -> None:
        """Update world model on each transition (test-time training)."""
        error = self.planner.update_on_transition(before, action, after)
        self._wm_errors.append(error)

    def _is_wm_reliable(self) -> bool:
        """Check if WM has learned enough to trust for planning."""
        if len(self._wm_errors) < 10:
            return False
        recent = self._wm_errors[-10:]
        avg_error = sum(recent) / len(recent)
        return avg_error < 0.1

    def _explore_action(self, frame: FrameDataRaw, actions: list[int]) -> int:
        """Policy-guided exploration with curiosity."""
        z = self.core.encode(frame_to_model_input(frame))
        
        with torch.no_grad():
            action_id = self.core.policy.select_action(z, actions, temperature=0.8)
        
        return action_id
