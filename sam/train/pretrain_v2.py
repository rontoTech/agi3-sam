"""SAM v2 Training Pipeline: Multi-objective pretraining for ARC-AGI-3.

Training objectives:
1. World Model (next-state prediction): z' = WM(z, a) 
2. Inverse Dynamics (action prediction): a = ID(z, z')
3. Temporal Contrastive: same-game states closer than cross-game states
4. Progress Prediction: predict if action leads to level-up
5. Action-Value: predict cumulative reward for (state, action) pairs

Data collection uses smart exploration (not just random actions):
- Momentum-based exploration (continue productive directions)
- Reset-and-retry for game overs
- Multi-seed for diversity
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from sam.core.sam_core import SamCore
from sam.train.checkpoint import save_sam_checkpoint
from sam.utils import frame_to_model_input

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single state transition from gameplay."""
    game_id: str
    frame_before: list
    action_id: int
    frame_after: list
    levels_before: int
    levels_after: int
    is_game_over: bool
    step_idx: int


@dataclass
class GameEpisode:
    """A full episode of gameplay."""
    game_id: str
    transitions: list[Transition] = field(default_factory=list)
    levels_achieved: int = 0
    total_steps: int = 0


class SmartDataCollector:
    """Collect diverse training data using intelligent exploration."""

    def __init__(self, environments_dir: str = "environment_files") -> None:
        self.environments_dir = environments_dir
        self.arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=environments_dir,
        )

    def collect_game(
        self,
        game_id: str,
        max_steps: int = 500,
        num_episodes: int = 5,
    ) -> list[GameEpisode]:
        """Collect multiple episodes with different exploration strategies."""
        episodes = []
        strategies = [
            self._momentum_explore,
            self._random_explore,
            self._systematic_explore,
            self._aggressive_explore,
            self._cautious_explore,
        ]
        
        for ep_idx in range(num_episodes):
            strategy = strategies[ep_idx % len(strategies)]
            episode = self._run_episode(game_id, max_steps, strategy)
            if episode.transitions:
                episodes.append(episode)
        
        return episodes

    def _run_episode(self, game_id: str, max_steps: int, strategy) -> GameEpisode:
        """Run a single episode with given exploration strategy."""
        env = self.arcade.make(game_id, save_recording=False)
        if env is None:
            return GameEpisode(game_id=game_id)
        
        frame = env.reset()
        if frame is None:
            return GameEpisode(game_id=game_id)

        episode = GameEpisode(game_id=game_id)
        prev_actions: list[int] = []
        
        for step in range(max_steps):
            if frame.state == GameState.WIN:
                break
            
            avail = [a for a in frame.available_actions if a != 0]
            if not avail:
                break
            
            aid = strategy(avail, prev_actions, step)
            
            action = GameAction.from_id(aid)
            if action.is_simple():
                action.action_data.game_id = game_id
            elif action.is_complex():
                x = random.randint(0, 63)
                y = random.randint(0, 63)
                action.set_data({"game_id": game_id, "x": x, "y": y})
            
            next_frame = env.step(action)
            if next_frame is None:
                break
            
            transition = Transition(
                game_id=game_id,
                frame_before=frame_to_model_input(frame),
                action_id=aid,
                frame_after=frame_to_model_input(next_frame),
                levels_before=frame.levels_completed,
                levels_after=next_frame.levels_completed,
                is_game_over=(next_frame.state == GameState.GAME_OVER),
                step_idx=step,
            )
            episode.transitions.append(transition)
            prev_actions.append(aid)
            
            if next_frame.state == GameState.GAME_OVER:
                action = GameAction.from_id(GameAction.RESET.value)
                action.action_data.game_id = game_id
                frame = env.step(action)
                if frame is None:
                    break
                prev_actions = []
                continue
            
            episode.levels_achieved = max(episode.levels_achieved, next_frame.levels_completed)
            frame = next_frame
        
        episode.total_steps = len(episode.transitions)
        return episode

    def _momentum_explore(self, avail: list[int], prev: list[int], step: int) -> int:
        """Continue last action with momentum, switch periodically."""
        if prev and prev[-1] in avail:
            run = sum(1 for a in reversed(prev) if a == prev[-1])
            if run < 5:
                return prev[-1]
        return random.choice(avail)

    def _random_explore(self, avail: list[int], prev: list[int], step: int) -> int:
        """Uniformly random action selection."""
        return random.choice(avail)

    def _systematic_explore(self, avail: list[int], prev: list[int], step: int) -> int:
        """Cycle through actions systematically."""
        return avail[step % len(avail)]

    def _aggressive_explore(self, avail: list[int], prev: list[int], step: int) -> int:
        """Prefer less-used actions."""
        if not prev:
            return random.choice(avail)
        counts = defaultdict(int)
        for a in prev[-20:]:
            counts[a] += 1
        min_count = min(counts.get(a, 0) for a in avail)
        least_used = [a for a in avail if counts.get(a, 0) == min_count]
        return random.choice(least_used)

    def _cautious_explore(self, avail: list[int], prev: list[int], step: int) -> int:
        """Move slowly, avoid repeating recent patterns."""
        if len(prev) >= 4 and prev[-1] == prev[-2] == prev[-3] == prev[-4]:
            others = [a for a in avail if a != prev[-1]]
            if others:
                return random.choice(others)
        return random.choice(avail)


class MultiObjectiveTrainer:
    """Train SAM core with multiple learning objectives."""

    def __init__(
        self,
        core: SamCore,
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.core = core
        self.device = torch.device(device)
        self.core.to(self.device)
        
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(core.latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 8),
        ).to(self.device)
        
        self.progress_head = nn.Sequential(
            nn.Linear(core.latent_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)
        
        all_params = (
            list(self.core.parameters())
            + list(self.inverse_dynamics.parameters())
            + list(self.progress_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
        
        self.train_stats: dict[str, list[float]] = defaultdict(list)

    def train_epoch(self, episodes: list[GameEpisode], batch_size: int = 32) -> dict[str, float]:
        """Train one epoch on collected episodes."""
        self.core.train()
        
        all_transitions = []
        for ep in episodes:
            all_transitions.extend(ep.transitions)
        
        if not all_transitions:
            return {"total_loss": 0.0}
        
        random.shuffle(all_transitions)
        
        total_loss = 0.0
        wm_loss_sum = 0.0
        id_loss_sum = 0.0
        progress_loss_sum = 0.0
        contrastive_loss_sum = 0.0
        value_loss_sum = 0.0
        n_batches = 0
        
        for i in range(0, len(all_transitions), batch_size):
            batch = all_transitions[i:i + batch_size]
            if len(batch) < 4:
                continue
            
            loss, losses = self._train_batch(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.core.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            wm_loss_sum += losses["wm"]
            id_loss_sum += losses["id"]
            progress_loss_sum += losses["progress"]
            contrastive_loss_sum += losses["contrastive"]
            value_loss_sum += losses["value"]
            n_batches += 1
        
        if n_batches == 0:
            return {"total_loss": 0.0}
        
        stats = {
            "total_loss": total_loss / n_batches,
            "wm_loss": wm_loss_sum / n_batches,
            "id_loss": id_loss_sum / n_batches,
            "progress_loss": progress_loss_sum / n_batches,
            "contrastive_loss": contrastive_loss_sum / n_batches,
            "value_loss": value_loss_sum / n_batches,
        }
        for k, v in stats.items():
            self.train_stats[k].append(v)
        return stats

    def _train_batch(self, batch: list[Transition]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute all training losses for a batch."""
        frames_before = [t.frame_before for t in batch]
        frames_after = [t.frame_after for t in batch]
        actions = [t.action_id for t in batch]
        level_ups = [1.0 if t.levels_after > t.levels_before else 0.0 for t in batch]
        game_overs = [1.0 if t.is_game_over else 0.0 for t in batch]
        
        z_before = self.core.encoder.encode_batch(frames_before)
        z_after = self.core.encoder.encode_batch(frames_after)
        
        wm_loss = self._world_model_loss(z_before, z_after, actions)
        id_loss = self._inverse_dynamics_loss(z_before, z_after, actions)
        progress_loss = self._progress_loss(z_before, actions, level_ups)
        contrastive_loss = self._contrastive_loss(z_before, z_after, batch)
        value_loss = self._value_loss(z_before, level_ups, game_overs)
        
        total = (
            1.0 * wm_loss
            + 0.5 * id_loss
            + 0.3 * progress_loss
            + 0.2 * contrastive_loss
            + 0.3 * value_loss
        )
        
        return total, {
            "wm": wm_loss.item(),
            "id": id_loss.item(),
            "progress": progress_loss.item(),
            "contrastive": contrastive_loss.item(),
            "value": value_loss.item(),
        }

    def _world_model_loss(
        self, z_before: torch.Tensor, z_after: torch.Tensor, actions: list[int]
    ) -> torch.Tensor:
        """Next-state prediction loss."""
        losses = []
        for i, aid in enumerate(actions):
            z_pred, conf = self.core.world_model(z_before[i], aid)
            target = z_after[i].detach()
            if z_pred.dim() == 1:
                z_pred = z_pred.unsqueeze(0)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            losses.append(F.mse_loss(z_pred, target))
        return torch.stack(losses).mean()

    def _inverse_dynamics_loss(
        self, z_before: torch.Tensor, z_after: torch.Tensor, actions: list[int]
    ) -> torch.Tensor:
        """Predict action from (z, z') pair."""
        z_cat = torch.cat([z_before, z_after.detach()], dim=-1)
        logits = self.inverse_dynamics(z_cat)
        targets = torch.tensor(actions, device=self.device, dtype=torch.long)
        return F.cross_entropy(logits, targets)

    def _progress_loss(
        self, z_before: torch.Tensor, actions: list[int], level_ups: list[float]
    ) -> torch.Tensor:
        """Predict if action leads to progress."""
        action_embeds = self.core.world_model.action_embed(
            torch.tensor(actions, device=self.device)
        )
        combined = torch.cat([z_before, action_embeds], dim=-1)
        pred = self.progress_head(combined).squeeze(-1)
        targets = torch.tensor(level_ups, device=self.device, dtype=torch.float32)
        return F.binary_cross_entropy(pred, targets)

    def _contrastive_loss(
        self, z_before: torch.Tensor, z_after: torch.Tensor, batch: list[Transition]
    ) -> torch.Tensor:
        """Temporal contrastive: consecutive states should be closer than random states."""
        if z_before.shape[0] < 4:
            return torch.tensor(0.0, device=self.device)
        
        positives = F.cosine_similarity(z_before, z_after, dim=-1)
        
        perm = torch.randperm(z_after.shape[0], device=self.device)
        negatives = F.cosine_similarity(z_before, z_after[perm], dim=-1)
        
        margin = 0.5
        loss = F.relu(negatives - positives + margin).mean()
        return loss

    def _value_loss(
        self, z: torch.Tensor, level_ups: list[float], game_overs: list[float]
    ) -> torch.Tensor:
        """Train value head to predict rewards."""
        scalar_v, _ = self.core.value(z)
        
        targets = torch.tensor(
            [5.0 * lu - 2.0 * go for lu, go in zip(level_ups, game_overs)],
            device=self.device, dtype=torch.float32,
        )
        return F.mse_loss(scalar_v.squeeze(), targets)


def pretrain_sam(
    config_path: Path = Path("sam/configs/global.yaml"),
    output_path: Path = Path("sam/checkpoints/global_sam.pt"),
    epochs: int = 10,
    steps_per_game: int = 500,
    episodes_per_game: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Full pretraining pipeline."""
    from sam.config import load_yaml_config
    
    cfg = load_yaml_config(config_path)
    env_dir = cfg.get("environments_dir", "environment_files")
    games = cfg.get("games", ["ls20-9607627b"])
    latent_dim = int(cfg.get("latent_dim", 128))
    lr = float(cfg.get("learning_rate", 3e-4))
    
    logger.info("Pretraining SAM on %d games, %d epochs", len(games), epochs)
    
    core = SamCore(latent_dim=latent_dim)
    trainer = MultiObjectiveTrainer(core, lr=lr, device=device)
    
    logger.info("Collecting training data...")
    collector = SmartDataCollector(env_dir)
    all_episodes: list[GameEpisode] = []
    
    for game_id in games:
        try:
            episodes = collector.collect_game(
                game_id,
                max_steps=steps_per_game,
                num_episodes=episodes_per_game,
            )
            total_trans = sum(len(ep.transitions) for ep in episodes)
            max_level = max((ep.levels_achieved for ep in episodes), default=0)
            logger.info(
                "  %s: %d episodes, %d transitions, max_level=%d",
                game_id, len(episodes), total_trans, max_level
            )
            all_episodes.extend(episodes)
        except Exception as e:
            logger.warning("  %s: failed - %s", game_id, e)
    
    total_transitions = sum(len(ep.transitions) for ep in all_episodes)
    logger.info("Total: %d episodes, %d transitions", len(all_episodes), total_transitions)
    
    logger.info("Training...")
    best_loss = float("inf")
    for epoch in range(epochs):
        stats = trainer.train_epoch(all_episodes, batch_size=batch_size)
        loss = stats["total_loss"]
        if loss < best_loss:
            best_loss = loss
            save_sam_checkpoint(core, output_path)
        logger.info(
            "Epoch %d/%d: loss=%.4f (wm=%.4f id=%.4f prog=%.4f contr=%.4f val=%.4f)",
            epoch + 1, epochs, loss,
            stats["wm_loss"], stats["id_loss"], stats["progress_loss"],
            stats["contrastive_loss"], stats["value_loss"],
        )
    
    save_sam_checkpoint(core, output_path)
    
    report = {
        "games": len(games),
        "episodes": len(all_episodes),
        "transitions": total_transitions,
        "epochs": epochs,
        "best_loss": best_loss,
        "params": core.total_params(),
        "checkpoint": str(output_path),
    }
    
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Training complete. Checkpoint: %s", output_path)
    
    return report
