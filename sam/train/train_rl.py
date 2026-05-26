"""Reinforcement Learning training for SAM policy.

Trains the policy network using REINFORCE with baseline to learn
game-solving strategies. The key difference from supervised pretraining:
- The policy learns FROM REWARDS (level completions, progress)
- It learns STRATEGIES (sequences of actions that achieve goals)
- It handles the exploration-exploitation tradeoff

Training loop:
1. Collect episodes using current policy (with exploration noise)
2. Compute returns (discounted rewards)
3. Update policy using policy gradient (REINFORCE with baseline)
4. Update value function as baseline
5. Optionally update WM from observed transitions

The policy is trained to maximize expected sum of rewards where:
- Level completion: +10
- GAME_OVER: -5
- Step penalty: -0.01 (encourages efficiency)
- Novel state visit: +0.1
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from sam.core.sam_core import SamCore
from sam.train.checkpoint import save_sam_checkpoint
from sam.utils import frame_hash, frame_to_model_input

logger = logging.getLogger(__name__)


@dataclass
class RLTransition:
    """Single step in an RL episode."""
    frame: list
    action: int
    reward: float
    next_frame: list
    done: bool
    log_prob: float
    value: float


@dataclass
class RLEpisode:
    """Full RL episode."""
    game_id: str
    transitions: list[RLTransition] = field(default_factory=list)
    total_reward: float = 0.0
    levels_completed: int = 0
    win: bool = False


class PolicyGradientTrainer:
    """REINFORCE with baseline for SAM policy training.
    
    This trainer collects gameplay episodes using the current policy,
    computes advantages, and updates the policy to increase probability
    of actions that led to good outcomes.
    """

    def __init__(
        self,
        core: SamCore,
        lr_policy: float = 1e-4,
        lr_value: float = 3e-4,
        lr_wm: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.core = core
        self.device = torch.device(device)
        self.core.to(self.device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.policy_optimizer = torch.optim.Adam(
            list(self.core.policy.parameters()) +
            list(self.core.encoder.parameters()),
            lr=lr_policy,
        )
        self.value_optimizer = torch.optim.Adam(
            self.core.value.parameters(), lr=lr_value
        )
        self.wm_optimizer = torch.optim.Adam(
            self.core.world_model.parameters(), lr=lr_wm
        )

    def collect_episode(
        self,
        game_id: str,
        env_dir: str = "environment_files",
        max_steps: int = 500,
        temperature: float = 1.0,
    ) -> RLEpisode:
        """Collect one episode using current policy."""
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=env_dir,
        )
        env = arcade.make(game_id, save_recording=False)
        if env is None:
            return RLEpisode(game_id=game_id)
        
        frame = env.reset()
        if frame is None:
            return RLEpisode(game_id=game_id)
        
        episode = RLEpisode(game_id=game_id)
        visited_hashes: set[str] = set()
        
        self.core.eval()
        
        for step in range(max_steps):
            if frame.state == GameState.WIN:
                episode.win = True
                break
            
            avail = [a for a in frame.available_actions if a != 0]
            if not avail:
                break
            
            with torch.no_grad():
                z = self.core.encode(frame_to_model_input(frame))
                logits = self.core.policy.masked_logits(z, avail)
                probs = F.softmax(logits / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_tensor = dist.sample()
                action_id = int(action_tensor.item())
                log_prob = float(dist.log_prob(action_tensor).item())
                
                scalar_v, _ = self.core.value(z)
                value = float(scalar_v.item())
            
            action = GameAction.from_id(action_id)
            if action.is_simple():
                action.action_data.game_id = game_id
            elif action.is_complex():
                action.set_data({"game_id": game_id, "x": 32, "y": 32})
            
            next_frame = env.step(action)
            if next_frame is None:
                break
            
            reward = self._compute_reward(
                frame, next_frame, action_id, visited_hashes
            )
            
            h = frame_hash(next_frame)
            visited_hashes.add(h)
            
            done = next_frame.state in (GameState.WIN, GameState.GAME_OVER)
            
            episode.transitions.append(RLTransition(
                frame=frame_to_model_input(frame),
                action=action_id,
                reward=reward,
                next_frame=frame_to_model_input(next_frame),
                done=done,
                log_prob=log_prob,
                value=value,
            ))
            episode.total_reward += reward
            
            if next_frame.levels_completed > frame.levels_completed:
                episode.levels_completed = next_frame.levels_completed
            
            if next_frame.state == GameState.GAME_OVER:
                action = GameAction.from_id(GameAction.RESET.value)
                action.action_data.game_id = game_id
                frame = env.step(action)
                if frame is None:
                    break
            else:
                frame = next_frame
        
        return episode

    def _compute_reward(
        self,
        frame: FrameDataRaw,
        next_frame: FrameDataRaw,
        action_id: int,
        visited: set[str],
    ) -> float:
        """Compute reward signal for RL training."""
        reward = -0.01
        
        if next_frame.levels_completed > frame.levels_completed:
            reward += 10.0
        
        if next_frame.state == GameState.GAME_OVER:
            reward -= 5.0
        
        h = frame_hash(next_frame)
        if h not in visited:
            reward += 0.1
        
        return reward

    def update_from_episodes(self, episodes: list[RLEpisode]) -> dict[str, float]:
        """Update policy and value from collected episodes."""
        if not episodes or all(not ep.transitions for ep in episodes):
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        all_returns = []
        all_log_probs = []
        all_values = []
        all_advantages = []
        all_frames = []
        all_actions = []
        all_next_frames = []
        
        for ep in episodes:
            if not ep.transitions:
                continue
            
            returns = self._compute_returns(ep)
            
            for i, (trans, ret) in enumerate(zip(ep.transitions, returns)):
                advantage = ret - trans.value
                all_returns.append(ret)
                all_log_probs.append(trans.log_prob)
                all_values.append(trans.value)
                all_advantages.append(advantage)
                all_frames.append(trans.frame)
                all_actions.append(trans.action)
                all_next_frames.append(trans.next_frame)
        
        if not all_returns:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        advantages = torch.tensor(all_advantages, dtype=torch.float32)
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.core.train()
        
        policy_loss = self._update_policy(all_frames, all_actions, advantages)
        value_loss = self._update_value(all_frames, all_returns)
        wm_loss = self._update_wm(all_frames, all_actions, all_next_frames)
        
        self.core.eval()
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "wm_loss": wm_loss,
            "avg_return": sum(all_returns) / len(all_returns),
            "avg_advantage": float(advantages.mean()),
        }

    def _compute_returns(self, episode: RLEpisode) -> list[float]:
        """Compute discounted returns for an episode."""
        returns = []
        G = 0.0
        for trans in reversed(episode.transitions):
            G = trans.reward + self.gamma * G * (0.0 if trans.done else 1.0)
            returns.insert(0, G)
        return returns

    def _update_policy(
        self, frames: list, actions: list[int], advantages: torch.Tensor
    ) -> float:
        """Policy gradient update."""
        batch_size = min(64, len(frames))
        total_loss = 0.0
        n_batches = 0
        
        indices = list(range(len(frames)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_frames = [frames[j] for j in batch_idx]
            batch_actions = [actions[j] for j in batch_idx]
            batch_adv = advantages[batch_idx].to(self.device)
            
            z = self.core.encoder.encode_batch(batch_frames)
            
            log_probs = []
            entropies = []
            for j, aid in enumerate(batch_actions):
                logits = self.core.policy.logits(z[j:j+1]).squeeze(0)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs.append(dist.log_prob(torch.tensor(aid, device=self.device)))
                entropies.append(dist.entropy())
            
            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies)
            
            policy_loss = -(log_probs_t * batch_adv.detach()).mean()
            entropy_loss = -self.entropy_coef * entropy_t.mean()
            
            loss = policy_loss + entropy_loss
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.core.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)

    def _update_value(self, frames: list, returns: list[float]) -> float:
        """Value function update."""
        batch_size = min(64, len(frames))
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_returns = returns[i:i + batch_size]
            
            z = self.core.encoder.encode_batch(batch_frames)
            scalar_v, _ = self.core.value(z)
            
            targets = torch.tensor(batch_returns, device=self.device, dtype=torch.float32)
            loss = F.mse_loss(scalar_v.squeeze(-1), targets)
            
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)

    def _update_wm(self, frames: list, actions: list[int], next_frames: list) -> float:
        """World model update from RL data."""
        batch_size = min(64, len(frames))
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(frames), batch_size):
            bf = frames[i:i + batch_size]
            ba = actions[i:i + batch_size]
            bnf = next_frames[i:i + batch_size]
            
            z = self.core.encoder.encode_batch(bf)
            z_next = self.core.encoder.encode_batch(bnf).detach()
            
            wm_loss = torch.tensor(0.0, device=self.device)
            for j, aid in enumerate(ba):
                pred, _ = self.core.world_model(z[j], aid)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                wm_loss = wm_loss + F.mse_loss(pred, z_next[j:j+1])
            wm_loss = wm_loss / len(ba)
            
            self.wm_optimizer.zero_grad()
            wm_loss.backward()
            self.wm_optimizer.step()
            
            total_loss += wm_loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)


def train_rl(
    config_path: Path = Path("sam/configs/global.yaml"),
    checkpoint_path: Path = Path("sam/checkpoints/global_sam.pt"),
    output_path: Path = Path("sam/checkpoints/global_sam_rl.pt"),
    num_iterations: int = 50,
    episodes_per_iter: int = 10,
    max_steps: int = 300,
    device: str = "cpu",
) -> dict[str, Any]:
    """Full RL training loop."""
    from sam.config import load_yaml_config
    
    cfg = load_yaml_config(config_path)
    env_dir = cfg.get("environments_dir", "environment_files")
    games = cfg.get("games", ["ls20-9607627b"])
    latent_dim = int(cfg.get("latent_dim", 128))
    
    core = SamCore(latent_dim=latent_dim)
    if checkpoint_path.exists():
        from sam.train.checkpoint import load_sam_checkpoint
        load_sam_checkpoint(core, checkpoint_path, device=device)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    
    trainer = PolicyGradientTrainer(core, device=device)
    
    best_avg_levels = 0.0
    history = []
    
    for iteration in range(num_iterations):
        all_episodes = []
        game_id = games[iteration % len(games)]
        
        for _ in range(episodes_per_iter):
            temperature = max(0.5, 1.0 - iteration * 0.01)
            ep = trainer.collect_episode(
                game_id, env_dir, max_steps=max_steps, temperature=temperature
            )
            if ep.transitions:
                all_episodes.append(ep)
        
        if not all_episodes:
            continue
        
        stats = trainer.update_from_episodes(all_episodes)
        
        avg_levels = sum(ep.levels_completed for ep in all_episodes) / len(all_episodes)
        avg_reward = sum(ep.total_reward for ep in all_episodes) / len(all_episodes)
        wins = sum(1 for ep in all_episodes if ep.win)
        
        if avg_levels > best_avg_levels:
            best_avg_levels = avg_levels
            save_sam_checkpoint(core, output_path)
        
        iteration_stats = {
            "iteration": iteration + 1,
            "game": game_id,
            "avg_levels": avg_levels,
            "avg_reward": avg_reward,
            "wins": wins,
            **stats,
        }
        history.append(iteration_stats)
        
        if (iteration + 1) % 5 == 0:
            logger.info(
                "Iter %d/%d [%s]: levels=%.1f reward=%.1f wins=%d ploss=%.4f vloss=%.4f",
                iteration + 1, num_iterations, game_id,
                avg_levels, avg_reward, wins,
                stats.get("policy_loss", 0),
                stats.get("value_loss", 0),
            )
    
    save_sam_checkpoint(core, output_path)
    
    report = {
        "iterations": num_iterations,
        "best_avg_levels": best_avg_levels,
        "games_trained": len(set(games)),
        "checkpoint": str(output_path),
        "history": history[-10:],
    }
    
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    
    return report
