"""SAM v2 Agent: Sophisticated AI agent for ARC-AGI-3.

Combines:
1. Pretrained world model + encoder (multi-objective learning on 25 games)
2. MCTS-style planning (think before acting)
3. Test-time training (adapt to each game online)
4. Adaptive exploration-exploitation balance
5. Graph-based state memory for efficient navigation
6. Reward-driven learning with 7-dim intrinsic motivation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from arc_agi import Arcade, OperationMode
from arc_agi.wrapper import EnvironmentWrapper
from arcengine import FrameDataRaw, GameAction, GameState

from sam.audit.logger import AuditLogger
from sam.audit.rhae import compute_rhae_report
from sam.config import GameMetadata, find_metadata, load_yaml_config
from sam.core.sam_core import SamCore
from sam.planner.mcts_planner import AdaptivePlanner, WorldModelPlanner
from sam.planner.state_graph import StateGraph
from sam.rewards.engine import RewardVectorEngine
from sam.train.checkpoint import load_sam_checkpoint, save_sam_checkpoint
from sam.utils import frame_hash, frame_to_model_input, raw_to_frame_data

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    """Per-episode statistics."""
    env_steps: int = 0
    level_actions: list[int] = field(default_factory=list)
    current_level_steps: int = 0
    resets: int = 0
    planning_actions: int = 0
    explore_actions: int = 0
    wm_errors: list[float] = field(default_factory=list)
    levels_completed: int = 0


class SamV2Agent:
    """SAM v2: Neural agent with world model planning and test-time learning.
    
    Architecture:
    - GridEncoder: 64×64 → latent z (128-dim)
    - WorldModel: p(z'|z,a) for internal simulation
    - MCTS Planner: searches over action sequences using WM
    - ValueHead: evaluates states for planning
    - PolicyHead: guides exploration + provides action priors for MCTS
    - SkillBank: grows capacity on new challenges
    - RewardEngine: 7-dim intrinsic motivation
    
    Test-time learning:
    - Every real transition updates WM to reduce prediction error
    - Value head learns from observed rewards
    - Policy improves from MCTS visit counts (expert iteration)
    
    The agent gets BETTER as it plays each game, using the first levels
    to learn game-specific dynamics that help with harder levels.
    """

    def __init__(
        self,
        game_id: str,
        environments_dir: str | Path = "environment_files",
        config_path: str | Path | None = None,
        checkpoint: str | Path | None = None,
        run_dir: str | Path = "sam/runs",
        planning_ms: float = 50.0,
        num_simulations: int = 32,
        device: str = "cpu",
    ) -> None:
        self.game_id = game_id
        self.environments_dir = Path(environments_dir)
        
        cfg_path = config_path or "sam/configs/ls20.yaml"
        self.config = load_yaml_config(Path(cfg_path))
        
        self.metadata = find_metadata(game_id, self.environments_dir)
        if self.metadata is None:
            raise FileNotFoundError(f"No metadata for {game_id}")
        
        self.device = torch.device(device)
        latent_dim = int(self.config.get("latent_dim", 128))
        self.core = SamCore(latent_dim=latent_dim).to(self.device)
        
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if ckpt_path.exists():
                load_sam_checkpoint(self.core, ckpt_path, device=self.device)
                logger.info("Loaded checkpoint: %s", checkpoint)
        
        self.planner = AdaptivePlanner(
            self.core,
            planning_budget_ms=planning_ms,
            num_simulations=num_simulations,
            device=device,
        )
        self.graph = StateGraph(max_depth=int(self.config.get("max_depth", 20)))
        self.rewards = RewardVectorEngine()
        
        self.run_dir = Path(run_dir)
        self.audit = AuditLogger(self.run_dir, game_id)
        
        self.stats = AgentStats()
        self._ttt_optimizer: torch.optim.Optimizer | None = None
        self._replay_buffer: list[dict] = []

    def run(self) -> dict[str, Any]:
        """Run the agent on the game."""
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(self.environments_dir),
        )
        env = arcade.make(self.game_id, save_recording=False)
        if env is None:
            raise RuntimeError(f"Cannot create env for {self.game_id}")
        
        frame = env.reset()
        if frame is None:
            raise RuntimeError("Reset failed")
        
        logger.info("Starting SAM v2 on %s", self.game_id)
        logger.info("Params: %d (budget: 3M)", self.core.total_params())
        
        self.stats = AgentStats()
        self.planner.reset_level()
        self.rewards.reset_level()
        
        prev_frame: FrameDataRaw | None = None
        
        while frame.state not in (GameState.WIN,):
            level = frame.levels_completed
            max_actions = self.metadata.max_actions_for_level(level)
            
            if self.stats.current_level_steps >= max_actions:
                logger.warning("Budget exhausted level %d", level)
                break
            
            if frame.state == GameState.GAME_OVER:
                frame = self._handle_game_over(env, frame)
                if frame is None:
                    break
                prev_frame = frame
                continue
            
            action_id, reason = self.planner.choose_action(frame, prev_frame)
            
            action = self._make_action(action_id)
            next_frame = env.step(action)
            if next_frame is None:
                break
            
            self.stats.env_steps += 1
            self.stats.current_level_steps += 1
            
            if reason == "mcts":
                self.stats.planning_actions += 1
            else:
                self.stats.explore_actions += 1
            
            self._on_transition(frame, action_id, next_frame, reason)
            
            if next_frame.levels_completed > level:
                self._on_level_complete(next_frame, level)
            
            prev_frame = frame
            frame = next_frame
        
        win = frame.state == GameState.WIN
        if win and self.stats.current_level_steps > 0:
            self.stats.level_actions.append(self.stats.current_level_steps)
        
        self.stats.levels_completed = frame.levels_completed
        return self._build_result(win, frame)

    def _make_action(self, action_id: int) -> GameAction:
        """Create GameAction from action ID."""
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = self.game_id
        elif action.is_complex():
            action.set_data({"game_id": self.game_id, "x": 32, "y": 32})
        return action

    def _handle_game_over(
        self, env: EnvironmentWrapper, frame: FrameDataRaw
    ) -> FrameDataRaw | None:
        """Handle GAME_OVER by resetting."""
        action = self._make_action(GameAction.RESET.value)
        frame = env.step(action)
        if frame is None:
            return None
        self.stats.env_steps += 1
        self.stats.current_level_steps += 1
        self.stats.resets += 1
        self.planner.reset_level()
        self.rewards.reset_level()
        return frame

    def _on_transition(
        self,
        before: FrameDataRaw,
        action_id: int,
        after: FrameDataRaw,
        reason: str,
    ) -> None:
        """Process a transition: update graph, WM, rewards."""
        self.planner.on_transition(before, action_id, after)
        
        before_fd = raw_to_frame_data(before)
        after_fd = raw_to_frame_data(after)
        self.graph.add_transition(before_fd, action_id, after_fd)
        
        h_before = frame_hash(before)
        h_after = frame_hash(after)
        
        z = self.core.encode(frame_to_model_input(before))
        z_next = self.core.encode(frame_to_model_input(after))
        _, conf = self.core.world_model(z, action_id)
        _, val_before = self.core.value(z)
        _, val_after = self.core.value(z_next)
        
        rv = self.rewards.compute(
            state_hash=h_after,
            prev_hash=h_before,
            wm_confidence=float(conf.item()),
            value_delta=float(val_after.mean().item() - val_before.mean().item()),
            level_before=before.levels_completed,
            level_after=after.levels_completed,
            is_env_step=True,
            is_internal=False,
            game_over=(after.state == GameState.GAME_OVER),
            reset_after_over=False,
            grid_changed=0,
            action_id=action_id,
        )
        
        self._replay_buffer.append({
            "z": z.detach().cpu(),
            "z_next": z_next.detach().cpu(),
            "action": action_id,
            "reward": self.rewards.scalar(rv, None),
            "level_up": after.levels_completed > before.levels_completed,
        })
        
        self._test_time_update(z, z_next, action_id, rv)

    def _test_time_update(
        self, z: torch.Tensor, z_next: torch.Tensor, action_id: int, rv
    ) -> None:
        """Test-time training: update model from each real transition."""
        self.core.train()
        
        if self._ttt_optimizer is None:
            self._ttt_optimizer = torch.optim.Adam([
                {"params": self.core.world_model.parameters(), "lr": 5e-4},
                {"params": self.core.value.parameters(), "lr": 3e-4},
                {"params": self.core.policy.parameters(), "lr": 1e-4},
            ])
        
        z = z.detach()
        z_next = z_next.detach()
        
        z_pred, conf = self.core.world_model(z.squeeze(0), action_id)
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)
        if z_next.dim() == 1:
            z_next = z_next.unsqueeze(0)
        
        wm_loss = F.mse_loss(z_pred, z_next)
        
        scalar_v, _ = self.core.value(z)
        reward_target = torch.tensor(
            self.rewards.scalar(rv, None),
            device=self.device, dtype=torch.float32,
        )
        value_loss = F.mse_loss(scalar_v.squeeze(), reward_target)
        
        total_loss = wm_loss + 0.5 * value_loss
        
        self._ttt_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.core.parameters(), 1.0)
        self._ttt_optimizer.step()
        
        self.core.eval()
        self.stats.wm_errors.append(float(wm_loss.item()))

    def _on_level_complete(self, frame: FrameDataRaw, prev_level: int) -> None:
        """Handle level completion."""
        self.stats.level_actions.append(self.stats.current_level_steps)
        self.stats.current_level_steps = 0
        
        self.core.skills.on_level_success()
        
        self._meso_update()
        
        self.planner.reset_level()
        self.rewards.reset_level()
        
        logger.info(
            "Level %d complete in %d steps (RHAE: %.1f)",
            prev_level,
            self.stats.level_actions[-1],
            min(
                (self.metadata.baseline_for_level(prev_level) / self.stats.level_actions[-1]) ** 2 * 100,
                115.0,
            ) if self.stats.level_actions[-1] > 0 else 0,
        )

    def _meso_update(self) -> None:
        """Replay-based update after level completion."""
        if len(self._replay_buffer) < 20:
            return
        
        self.core.train()
        batch = self._replay_buffer[-100:]
        
        for item in batch:
            z = item["z"].to(self.device)
            z_next = item["z_next"].to(self.device)
            aid = item["action"]
            
            z_pred, _ = self.core.world_model(z.squeeze(0), aid)
            if z_pred.dim() == 1:
                z_pred = z_pred.unsqueeze(0)
            if z_next.dim() == 1:
                z_next = z_next.unsqueeze(0)
            
            loss = F.mse_loss(z_pred, z_next)
            
            if self._ttt_optimizer:
                self._ttt_optimizer.zero_grad()
                loss.backward()
                self._ttt_optimizer.step()
        
        self.core.eval()

    def _build_result(self, win: bool, frame: FrameDataRaw) -> dict[str, Any]:
        """Build result summary."""
        report = compute_rhae_report(self.metadata, self.stats.level_actions)
        
        avg_wm_error = (
            sum(self.stats.wm_errors[-50:]) / min(len(self.stats.wm_errors), 50)
            if self.stats.wm_errors else 0.0
        )
        
        return {
            "game_id": self.game_id,
            "win": win,
            "state": frame.state.name if hasattr(frame.state, "name") else str(frame.state),
            "levels_completed": self.stats.levels_completed,
            "env_steps": self.stats.env_steps,
            "resets": self.stats.resets,
            "planning_actions": self.stats.planning_actions,
            "explore_actions": self.stats.explore_actions,
            "modules": len(self.core.skills.modules),
            "total_params": self.core.total_params(),
            "avg_wm_error": round(avg_wm_error, 4),
            "rhae": report,
            "audit_path": str(self.audit.path),
        }
