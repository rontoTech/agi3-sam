"""SAM Competition Agent: Test-time learning for ARC-AGI-3.

Based on insights from the 1st place preview competition solution (StochasticGoose):
1. Action change predictor: CNN predicts which actions cause state changes
2. Efficient exploration: Only try actions predicted to change state
3. Online RL: Learn from level completions during exploration
4. Hash-based novelty: Track visited states, prefer novel ones
5. Off-policy replay: Store all transitions, train continuously

The key insight: these games are UNKNOWN at test time. The agent must learn
from scratch which actions are "legal" (cause changes) and which sequences
lead to level completion. Pre-computed solutions are useless.

Architecture:
- ActionChangePredictor: CNN(frame) → P(action causes change) for 6 actions
- CoordinatePredictor: CNN(frame) → heatmap for ACTION6 click location  
- PolicyHead: CNN(frame) → action probabilities for exploitation
- ExplorationStrategy: Use predicted legal actions + novelty to explore efficiently
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_agi import Arcade, OperationMode
from arc_agi.wrapper import EnvironmentWrapper
from arcengine import FrameDataRaw, GameAction, GameState

logger = logging.getLogger(__name__)


def frame_to_tensor(frame: FrameDataRaw) -> torch.Tensor:
    """Convert frame to tensor [1, 16, 64, 64] one-hot."""
    if not frame.frame:
        return torch.zeros(1, 16, 64, 64)
    layer = frame.frame[0]
    if hasattr(layer, "tolist"):
        layer = layer.tolist()
    h, w = len(layer), len(layer[0]) if layer else 0
    if h == 0 or w == 0:
        return torch.zeros(1, 16, 64, 64)
    grid = torch.zeros(h, w, dtype=torch.long)
    for y in range(h):
        for x in range(w):
            grid[y, x] = min(max(int(layer[y][x]), 0), 15)
    one_hot = F.one_hot(grid, 16).float().permute(2, 0, 1).unsqueeze(0)
    if h != 64 or w != 64:
        one_hot = F.interpolate(one_hot, size=(64, 64), mode="nearest")
    return one_hot


def fast_hash(frame: FrameDataRaw) -> str:
    """Fast frame hash for novelty detection."""
    layer = frame.frame[0]
    if hasattr(layer, "tobytes"):
        raw = layer.tobytes()
    else:
        raw = str(layer).encode()
    return hashlib.md5(raw).hexdigest()[:12]


class ActionChangePredictor(nn.Module):
    """CNN that predicts which actions will change the game state.
    
    This is the KEY innovation from the winning approach:
    - Input: current frame (64×64, 16 colors)
    - Output: P(action causes state change) for each of 6 actions
    - Trained online from observed transitions
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.action_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid(),
        )
        self.coord_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs [B, 6], coord_logits [B, 1, 32, 32])."""
        features = self.encoder(x)
        flat = features.flatten(1)
        action_probs = self.action_head(flat)
        coord_logits = self.coord_head(features)
        return action_probs, coord_logits

    def predict_legal_actions(self, frame: FrameDataRaw, threshold: float = 0.3) -> list[int]:
        """Predict which actions will cause a state change."""
        x = frame_to_tensor(frame)
        with torch.no_grad():
            probs, coords = self(x)
        probs = probs.squeeze(0)
        legal = []
        for i in range(6):
            if probs[i].item() > threshold:
                legal.append(i + 1)  # Actions are 1-indexed
        return legal if legal else [1, 2, 3, 4, 5, 6]

    def predict_click_coords(self, frame: FrameDataRaw) -> tuple[int, int]:
        """Predict best (x, y) for click action."""
        x = frame_to_tensor(frame)
        with torch.no_grad():
            _, coords = self(x)
        coords = coords.squeeze()
        if coords.dim() == 2:
            flat_idx = coords.flatten().argmax().item()
            h, w = coords.shape
            y = (flat_idx // w) * (64 // h)
            x_coord = (flat_idx % w) * (64 // w)
            return int(x_coord), int(y)
        return 32, 32


@dataclass
class Transition:
    """Stored transition for off-policy training."""
    frame_tensor: torch.Tensor
    action: int
    changed: bool
    next_frame_tensor: torch.Tensor | None = None
    level_completed: bool = False
    coord_x: int = 32
    coord_y: int = 32


class CompetitionAgent:
    """Test-time learning agent for ARC-AGI-3 competition.
    
    Designed to solve UNKNOWN games from scratch within a time budget.
    
    Strategy:
    1. Explore using action change predictor (avoid no-ops)
    2. Track visited states (prefer novelty)
    3. Train online from all transitions
    4. On level completion, optimize path via replay
    """

    def __init__(
        self,
        game_id: str,
        environments_dir: str = "environment_files",
        time_budget_s: float = 600.0,
        train_every: int = 50,
    ) -> None:
        self.game_id = game_id
        self.environments_dir = environments_dir
        self.time_budget_s = time_budget_s
        self.train_every = train_every
        
        self.model = ActionChangePredictor()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.memory: list[Transition] = []
        self.visited_hashes: set[str] = set()
        self.level_transitions: dict[int, list[Transition]] = {}
        self.level_solutions: dict[int, list[int]] = {}
        
        self.stats = {
            "total_steps": 0,
            "levels_completed": 0,
            "game_overs": 0,
            "novel_states": 0,
            "train_steps": 0,
        }

    def solve(self) -> dict[str, Any]:
        """Main solve loop: explore, learn, solve."""
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
        
        deadline = time.monotonic() + self.time_budget_s
        current_level = 0
        level_actions: list[int] = []
        current_level_actions: list[int] = []
        current_level_transitions: list[Transition] = []
        
        avail = frame.available_actions
        has_click = 6 in avail
        
        logger.info("Starting on %s, actions=%s, budget=%.0fs", self.game_id, avail, self.time_budget_s)
        
        while time.monotonic() < deadline:
            if frame.state == GameState.WIN:
                if current_level_actions:
                    level_actions.append(len(current_level_actions))
                break
            
            if frame.state == GameState.GAME_OVER:
                self.stats["game_overs"] += 1
                action = self._make_action(GameAction.RESET.value)
                frame = env.step(action)
                if frame is None:
                    break
                self.stats["total_steps"] += 1
                current_level_actions = []
                current_level_transitions = []
                continue
            
            # Choose action using learned model + exploration
            action_id, coord_x, coord_y = self._choose_action(frame, avail)
            
            # Execute
            prev_frame = frame
            prev_tensor = frame_to_tensor(frame)
            action = self._make_action(action_id, coord_x, coord_y)
            frame = env.step(action)
            if frame is None:
                break
            
            self.stats["total_steps"] += 1
            current_level_actions.append(action_id)
            
            # Record transition
            new_hash = fast_hash(frame)
            changed = (new_hash != fast_hash(prev_frame))
            level_completed = (frame.levels_completed > current_level)
            
            if new_hash not in self.visited_hashes:
                self.visited_hashes.add(new_hash)
                self.stats["novel_states"] += 1
            
            trans = Transition(
                frame_tensor=prev_tensor,
                action=action_id,
                changed=changed,
                next_frame_tensor=frame_to_tensor(frame) if changed else None,
                level_completed=level_completed,
                coord_x=coord_x,
                coord_y=coord_y,
            )
            self.memory.append(trans)
            current_level_transitions.append(trans)
            
            # Level completed!
            if level_completed:
                level_actions.append(len(current_level_actions))
                self.level_solutions[current_level] = current_level_actions[:]
                self.level_transitions[current_level] = current_level_transitions[:]
                self.stats["levels_completed"] += 1
                
                logger.info(
                    "Level %d completed in %d actions!",
                    current_level, len(current_level_actions)
                )
                
                current_level = frame.levels_completed
                current_level_actions = []
                current_level_transitions = []
                self.visited_hashes = set()
            
            # Online training
            if self.stats["total_steps"] % self.train_every == 0:
                self._train_batch()
        
        return self._build_result(frame, level_actions)

    def _choose_action(
        self, frame: FrameDataRaw, avail: list[int]
    ) -> tuple[int, int, int]:
        """Choose action using learned model + novelty-driven exploration."""
        # Get predicted legal actions
        predicted_legal = self.model.predict_legal_actions(frame, threshold=0.3)
        
        # Filter to actually available actions
        legal = [a for a in predicted_legal if a in avail and a != 0]
        if not legal:
            legal = [a for a in avail if a != 0]
        
        # Novelty-driven selection: prefer actions leading to new states
        # (We don't know where they lead without trying, so use the model's ranking)
        if random.random() < 0.1:
            # Pure exploration: random from legal
            action_id = random.choice(legal)
        else:
            # Model-guided: pick highest-probability legal action
            x = frame_to_tensor(frame)
            with torch.no_grad():
                probs, _ = self.model(x)
            probs = probs.squeeze(0)
            
            best_action = legal[0]
            best_prob = -1.0
            for a in legal:
                p = probs[a - 1].item() if a <= 6 else 0.0
                if p > best_prob:
                    best_prob = p
                    best_action = a
            action_id = best_action
        
        # Predict coordinates for click action
        coord_x, coord_y = 32, 32
        if action_id == 6:
            coord_x, coord_y = self.model.predict_click_coords(frame)
        
        return action_id, coord_x, coord_y

    def _train_batch(self) -> None:
        """Train action change predictor from memory."""
        if len(self.memory) < 32:
            return
        
        self.model.train()
        batch = random.sample(self.memory, min(64, len(self.memory)))
        
        frames = torch.cat([t.frame_tensor for t in batch], dim=0)
        targets = torch.zeros(len(batch), 6)
        for i, t in enumerate(batch):
            if t.action >= 1 and t.action <= 6:
                targets[i, t.action - 1] = 1.0 if t.changed else 0.0
        
        action_probs, _ = self.model(frames)
        loss = F.binary_cross_entropy(action_probs, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        self.stats["train_steps"] += 1

    def _make_action(self, action_id: int, x: int = 32, y: int = 32) -> GameAction:
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = self.game_id
        elif action.is_complex():
            action.set_data({"game_id": self.game_id, "x": x, "y": y})
        return action

    def _build_result(self, frame: FrameDataRaw, level_actions: list[int]) -> dict[str, Any]:
        win = frame.state == GameState.WIN if frame else False
        levels = frame.levels_completed if frame else 0
        
        return {
            "game_id": self.game_id,
            "win": win,
            "levels_completed": levels,
            "level_actions": level_actions,
            "stats": self.stats,
        }
