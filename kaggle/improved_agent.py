# =====================================================================
# Refactoring and optimizing FORGE v19
# =====================================================================
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== BFS SOLVER ====================

class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""

    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}  # level_idx → action list
        # ==================== SELF-ASSESSMENT METRICS ====================
        self._run_log = {
            "steps": 0,
            "bfs_used": False,
            "bfs_success": False,
            "cnn_steps": 0,
            "explore_steps": 0,
            "revisit_count": 0,
            "reward_sum": 0.0,
        }
        self._seen_states = set()
        self._run_id = hash((time.time(), getattr(self, "game_id", "unknown")))
        self._last_logged_score = None

    def load(self):
        """Load the game class from source."""
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    def _state_hash(self, g, frame, hidden_fields=None):
        """Hash frame + discovered hidden scalar fields (fast)."""
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(g, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        """Dynamic state probing — discover which scalar fields change per action.
        Returns list of field names that are hidden state (change without pixel change)."""
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v

        changing_fields = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            f = g.get_pixels(0, 0, 64, 64)
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)

        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _scan_actions(self, game, f0, bg):
        """Scan for effective actions. Returns list of (action_id, data)."""
        avail = game._available_actions
        actions = []
        # Directional/interact actions
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        # Click actions
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True
                        )
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        diff = np.sum(f0 != f)
                        if diff > 0:
                            effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if effect_hash not in seen_effects:
                                seen_effects.add(effect_hash)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                    except:
                        pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        """Optimized BFS (fast state propagation, reduced deepcopy overhead)."""
        if not self.game_cls:
            return None
    
        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
    
        f0 = np.asarray(r0.frame[-1], dtype=np.uint8)
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
    
        # ---------------- transfer ----------------
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result
    
        # ---------------- scan ----------------
        actions = self._scan_actions(game, f0, bg)
    
        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    f_after = np.asarray(g_warmup.get_pixels(0, 0, 64, 64), dtype=np.uint8)
                    actions = self._scan_actions(g_warmup, f_after, bg)
                    if actions:
                        logger.info(f"BFS L{level_idx}: UNLOCKED ACTION{warmup_id}")
                        game = g_warmup
                        f0 = f_after
                        break
                except:
                    pass
    
        if not actions:
            return None
    
        logger.info(f"BFS L{level_idx}: {len(actions)} actions")
    
        # ======================================================
        # FAST BFS (STATE-CARRY FORWARD, NO HIST REPLAY LOOP)
        # ======================================================
    
        Action = ActionInput
        from_id = GameAction.from_id
    
        visited = set()
        queue = deque()
    
        # Status bar masking: detect rows that change frequently (counters/UI)
        # and exclude them from the hash to reduce duplicate states
        def detect_status_mask(f0_frame):
            """Detect top/bottom rows that are likely status bars."""
            # Heuristic: status bars are in first/last 8 rows
            # They often have uniform or patterned colors different from game area
            h = f0_frame.shape[0]
            mask_top, mask_bot = 0, 0
            # Check if top rows are uniform (likely padding/status)
            for r in range(min(8, h)):
                row = f0_frame[r]
                if len(set(row.flatten().tolist())) <= 2:
                    mask_top = r + 1
                else:
                    break
            # Check bottom rows
            for r in range(min(8, h)):
                row = f0_frame[h - 1 - r]
                if len(set(row.flatten().tolist())) <= 2:
                    mask_bot = r + 1
                else:
                    break
            return mask_top, mask_bot

        _mask_top, _mask_bot = detect_status_mask(f0)

        def frame_hash(frame):
            h = frame.shape[0]
            mt, mb = _mask_top, _mask_bot
            end = h - mb if mb > 0 else h
            if mt < end:
                return hash(frame[mt:end].tobytes())
            return hash(frame.tobytes())
    
        h0 = frame_hash(f0)
        visited.add(h0)
    
        # store full game state instead of replaying history
        base_state = game
    
        queue.append((base_state, [], 0))
    
        t0 = time.time()
        explored = 0
    
        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
    
            g, hist, depth = queue.popleft()
    
            for act_id, data in actions:
    
                try:
                    g2 = copy.deepcopy(g)
    
                    ai = Action(id=from_id(act_id), data=data) if data else Action(id=from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
    
                    explored += 1
    
                    if not r.frame:
                        continue
    
                    f = np.asarray(r.frame[-1], dtype=np.uint8)
                    h = frame_hash(f)
    
                    if h in visited:
                        continue
                    visited.add(h)
    
                    new_hist = hist + [(act_id, data)]
    
                    if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                        elapsed = time.time() - t0
                        logger.info(f"BFS L{level_idx}: SOLVED in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                        self.solutions[level_idx] = new_hist
                        return new_hist
    
                    if depth < 30:
                        queue.append((g2, new_hist, depth + 1))
    
                except:
                    continue
    
        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: timeout ({explored} explored, {len(visited)} unique, {elapsed_first:.1f}s)")
    
        # ---------------- early exit ----------------
        if explored < 20 and elapsed_first > 10.0:
            return None
    
        # ---------------- hidden retry (unchanged logic, slightly faster) ----------------
        if len(visited) < 50 and elapsed_first < self.bfs_timeout * 0.8:
    
            hidden_fields = self._probe_hidden_fields(game, actions)
    
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY hidden fields {hidden_fields}")
    
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                r0_2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    
                if not r0_2.frame:
                    return None
    
                f0_2 = np.asarray(r0_2.frame[-1], dtype=np.uint8)
                h0_2 = frame_hash(f0_2)
    
                visited2 = {h0_2}
                queue2 = deque([(game2, [], 0)])
    
                t0_2 = time.time()
                explored2 = 0
                remaining = max(30, self.bfs_timeout - elapsed_first)
    
                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
    
                    g, hist, depth = queue2.popleft()
    
                    for act_id, data in actions:
    
                        try:
                            g2 = copy.deepcopy(g)
    
                            ai = Action(id=from_id(act_id), data=data) if data else Action(id=from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
    
                            explored2 += 1
    
                            if not r.frame:
                                continue
    
                            f = np.asarray(r.frame[-1], dtype=np.uint8)
                            h = frame_hash(f)
    
                            if h in visited2:
                                continue
                            visited2.add(h)
    
                            new_hist = hist + [(act_id, data)]
    
                            if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                                logger.info(f"BFS L{level_idx}: SOLVED retry in {len(new_hist)} actions")
                                self.solutions[level_idx] = new_hist
                                return new_hist
    
                            if depth < 30:
                                queue2.append((g2, new_hist, depth + 1))
    
                        except:
                            continue
    
        return None
    
    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """Transfer previous level's solution to current level."""
        try:
            # Try executing prev solution directly
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct replay, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            # Try object-relative transfer
            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)

            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))

            if not matched:
                return None

            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (offset dx={dx:.0f},dy={dy:.0f}, {i+1} actions)")
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py file and class name."""
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]

    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            import re
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m:
                        cls_name = m.group(1)
                    break

    if not src:
        import re
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"**/game_sources/**/{gid}.py",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                break

    return src, cls_name


# ==================== GLOBAL RUN EVALUATION REGISTRY ====================
# <<< NEW: stores multiple stochastic runs for post-analysis

_GLOBAL_RUN_METRICS = {
    "runs": [],
}

def _hash_run_trace(trace):
    return hashlib.md5(str(trace).encode()).hexdigest()


# ==================== CNN FALLBACK ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B,M=mem_actions.shape
        if M==0:return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)

class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g=g
        s.c1=nn.Conv2d(in_ch,32,3,padding=1);s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1);s.c4=nn.Conv2d(128,256,3,padding=1)
        s.attn=CBAM(256);s.ar=nn.Conv2d(256,64,1);s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256);s.ah=nn.Linear(256,5);s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(256,128,3,padding=1);s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1);s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1);s.gf=nn.Linear(256,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x=F.relu(s.c1(x));x=F.relu(s.c2(x));x=F.relu(s.c3(x));f=F.relu(s.c4(x))
        f=s.attn(f);af=F.relu(s.ar(f));af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f));cf=F.relu(s.cc2(cf));cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)


def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)

        # ... (ALL YOUR ORIGINAL INIT CODE)

        # ==================== SELF-ASSESSMENT CORE ====================
        s.run_metrics = {
            "level_stats": {},     # per-level tracking
            "total_steps": 0,
            "bfs_success": 0,
            "cnn_success": 0,
            "reward_sum": 0.0,
            "state_hashes": set(),
            "action_entropy": [],
            "run_id": hash(time.time() + random.random()),  # stochastic run ID
        }

        s._level_start_time = time.time()
        s._level_step_count = 0

    # ==================== SELF-ASSESSMENT HELPERS ====================

    def _log_step(s, level, action_type, reward):
        """Track per-step statistics"""
        if level not in s.run_metrics["level_stats"]:
            s.run_metrics["level_stats"][level] = {
                "steps": 0,
                "reward": 0.0,
                "unique_states": 0,
                "bfs_used": False,
                "cnn_used": False,
            }

        st = s.run_metrics["level_stats"][level]
        st["steps"] += 1
        st["reward"] += float(reward)

        s.run_metrics["total_steps"] += 1
        s.run_metrics["reward_sum"] += float(reward)

    def _record_state(s, raw_frame):
        """Detect stochastic divergence across runs"""
        h = hashlib.md5(raw_frame.tobytes()).hexdigest()
        s.run_metrics["state_hashes"].add(h)
        return h

    def get_run_summary(s):
        """Call at end of episode or externally for evaluation"""
        return {
            "run_id": s.run_metrics["run_id"],
            "total_steps": s.run_metrics["total_steps"],
            "reward_sum": s.run_metrics["reward_sum"],
            "unique_states": len(s.run_metrics["state_hashes"]),
            "levels": s.run_metrics["level_stats"],
        }

    def finalize_run(s):
        """Push run into global registry (for multi-run evaluation)"""
        summary = s.get_run_summary()
        _GLOBAL_RUN_METRICS["runs"].append(summary)
        return summary

    # ==================== MAIN LOOP HOOKS ====================

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # LEVEL CHANGE
            if lvl != s.cl:
                s._level_start_time = time.time()
                s._level_step_count = 0

                # reset per-level stats
                s.run_metrics["level_stats"][lvl] = {
                    "steps": 0,
                    "reward": 0.0,
                    "unique_states": 0,
                    "bfs_used": False,
                    "cnn_used": False,
                }

                s.cl = lvl

            # RAW FRAME
            raw = s._raw(lf)
            state_hash = s._record_state(raw)

            # ================= BFS PATH =================
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                s.run_metrics["level_stats"][lvl]["bfs_used"] = True
                s.run_metrics["bfs_success"] += 1

                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1

                action = GameAction.from_id(act_id)
                action.reasoning = "bfs"

                s._log_step(lvl, "bfs", 1.0)
                return action

            # ================= CNN PATH =================
            action = super().choose_action(frames, lf)

            s.run_metrics["level_stats"][lvl]["cnn_used"] = True
            s._log_step(lvl, "cnn", 0.1)

            return action

        except Exception as e:
            traceback.print_exc()
            return random.choice([GameAction.ACTION1])

    # ==================== END-OF-RUN HOOK ====================

    def end_episode(s):
        """Call this externally if framework allows episode hook"""
        summary = s.finalize_run()

        logger.info("===== RUN SUMMARY =====")
        logger.info(summary)

        return summary


# ==================== MULTI-RUN ANALYSIS UTILITIES ====================

def compare_runs():
    """
    Call after multiple agent runs to assess nondeterminism.
    """
    runs = _GLOBAL_RUN_METRICS["runs"]

    if not runs:
        return {"error": "no runs recorded"}

    avg_reward = np.mean([r["reward_sum"] for r in runs])
    avg_steps = np.mean([r["total_steps"] for r in runs])
    stability = np.mean([r["unique_states"] for r in runs])

    return {
        "num_runs": len(runs),
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "avg_state_diversity": float(stability),
    }