"""Q-learning library for LP/HP branch selection."""

import json
import os
import tempfile
import random
from typing import Optional, Dict, Any

EPSILON_MIN = 0.2
EPSILON_DECAY = 0.01
DEFAULT_EPSILON = 0.8


def _get_default_q_table() -> Dict[str, Any]:
    return {
        "calm": {"LP": 0.0, "HP": 0.0},
        "lively": {"LP": 0.0, "HP": 0.0},
        "epsilon": DEFAULT_EPSILON
    }


def _validate_q_table(data: Dict[str, Any]) -> Dict[str, Any]:
    for ctx in ["calm", "lively"]:
        if ctx not in data or not isinstance(data[ctx], dict):
            data[ctx] = {"LP": 0.0, "HP": 0.0}
        for branch in ["LP", "HP"]:
            if branch not in data[ctx]:
                data[ctx][branch] = 0.0
    
    if "epsilon" not in data:
        data["epsilon"] = DEFAULT_EPSILON
    else:
        eps = data["epsilon"]
        if not isinstance(eps, (int, float)):
            data["epsilon"] = DEFAULT_EPSILON
        else:
            data["epsilon"] = min(1.0, max(EPSILON_MIN, float(eps)))
    
    return data


def load_q_table(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        return _get_default_q_table()
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return _validate_q_table(data)
    except (json.JSONDecodeError, IOError, ValueError) as e:
        print(f"[learning] Warning: failed to load {json_path}: {e}")
        return _get_default_q_table()


def save_q_table(json_path: str, q: Dict[str, Any]) -> bool:
    try:
        q = _validate_q_table(q)
        dir_name = os.path.dirname(json_path) or '.'
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False) as tmp:
            json.dump(q, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, json_path)
        return True
    except (IOError, OSError) as e:
        print(f"[learning] Warning: failed to save {json_path}: {e}")
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass
        return False


def select_branch(context: int, q: Dict[str, Any], rng: random.Random) -> Optional[str]:
    """Epsilon-greedy branch selection. Returns None if context invalid."""
    if context not in (0, 1):
        return None
    
    ctx_key = "calm" if context == 0 else "lively"
    q_vals = q.get(ctx_key, {"LP": 0.0, "HP": 0.0})
    epsilon = q.get("epsilon", DEFAULT_EPSILON)
    
    # Exploration: random choice
    if rng.random() < epsilon:
        return rng.choice(["LP", "HP"])
    
    # Exploitation: choose best Q-value, randomize on tie
    lp_val = q_vals.get("LP", 0.0)
    hp_val = q_vals.get("HP", 0.0)
    if lp_val == hp_val:
        return rng.choice(["LP", "HP"])  # Random tie-breaking
    return "LP" if lp_val > hp_val else "HP"


def update_q(q: Dict[str, Any], context: int, branch: str, reward: float, eta: float) -> float:
    """Q ← Q + η(reward - Q). Mutates q in-place. Returns new Q-value."""
    if context not in (0, 1):
        print(f"[learning] Warning: invalid context {context}, skipping Q update")
        return 0.0
    
    ctx_key = "calm" if context == 0 else "lively"
    if ctx_key not in q:
        q[ctx_key] = {"LP": 0.0, "HP": 0.0}
    if branch not in q[ctx_key]:
        q[ctx_key][branch] = 0.0
    
    old_q = q[ctx_key][branch]
    new_q = old_q + eta * (reward - old_q)
    q[ctx_key][branch] = new_q
    return new_q


def decay_epsilon(q: Dict[str, Any]) -> float:
    """Decay epsilon with minimum threshold. Mutates q in-place."""
    current = q.get("epsilon", DEFAULT_EPSILON)
    new_eps = max(EPSILON_MIN, current - EPSILON_DECAY)
    q["epsilon"] = new_eps
    return new_eps


def compute_reward(v: float, a: float, alpha: float = 1.0, beta: float = 0.3) -> float:
    return alpha * v + beta * a


def clamp_reward(r: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, r))


def update_after_action(json_path: str, context: int, branch: str,
                        reward: float, eta: float = 0.1, decay_eps: bool = True) -> bool:
    """Atomic: load Q -> update Q -> decay epsilon -> save.
    
    Args:
        json_path: Path to Q-table JSON file
        context: Context (0=calm, 1=lively)
        branch: Branch taken ("LP" or "HP")
        reward: Reward value
        eta: Learning rate
        decay_eps: Whether to decay epsilon (False to skip decay on failed actions)
    """
    if context not in (0, 1):
        print(f"[learning] Invalid context {context}, skipping update")
        return False
    
    q = load_q_table(json_path)
    reward = clamp_reward(reward)
    new_q = update_q(q, context, branch, reward, eta)
    new_eps = q.get("epsilon", DEFAULT_EPSILON)
    
    if decay_eps:
        new_eps = decay_epsilon(q)
    
    success = save_q_table(json_path, q)
    
    if success:
        ctx_str = "calm" if context == 0 else "lively"
        decay_str = f", eps={new_eps:.2f}" if decay_eps else ""
        print(f"[learning] Updated Q[{ctx_str}][{branch}]={new_q:.3f}{decay_str}, r={reward:.3f}")
    
    return success
