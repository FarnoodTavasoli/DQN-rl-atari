"""
diagnose_fire.py — Check whether a trained agent presses FIRE when needed
==========================================================================
After a life loss (or at the start of a game) Breakout leaves the ball on
the paddle.  The agent *must* select the FIRE action to launch it.
FireResetEnv only injects FIRE on a true env.reset(); it does nothing after
an intermediate life loss handled by EpisodicLifeEnv.

This script:
  1. Loads a checkpoint.
  2. Runs several episodes in a *training-style* environment (EpisodicLife
     enabled) so that life-loss states are visible.
  3. Detects frames where the observation does not change (ball sitting on
     the paddle / game frozen waiting for launch).
  4. Records the action chosen by the policy in those states.
  5. Prints a clear summary: if FIRE is rarely selected, the diagnosis is
     confirmed.

Usage
-----
    python diagnose_fire.py --model_path checkpoints/ALE-Breakout-v5/DQN/002/DQN_final.zip --algorithm DQN --env_id ALE/Breakout-v5 --n_episodes 10

    # Also works with any intermediate checkpoint, e.g.:
    python diagnose_fire.py \\
        --model_path checkpoints/ALE-Breakout-v5/DoubleDQN/003/DoubleDQN_step_5000000_steps.zip \\
        --algorithm DoubleDQN
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from agent import DoubleDQN, DuelingCnnPolicy, DuelingDQN, StandardDQN
from environment import make_atari_env
from stable_baselines3.common.vec_env import VecEnv

# ---------------------------------------------------------------------------
# Algorithm registry (same as evaluate.py)
# ---------------------------------------------------------------------------
_ALGORITHM_REGISTRY = {
    "DQN": StandardDQN,
    "DoubleDQN": DoubleDQN,
    "DuelingDQN": DuelingDQN,
}


def load_model(model_path: str, algorithm: str, env: Optional[VecEnv] = None, device: str = "cuda"):
    """Load a checkpoint with the correct class."""
    if algorithm not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(_ALGORITHM_REGISTRY)}")
    AlgoClass = _ALGORITHM_REGISTRY[algorithm]
    custom_objects = {}
    if algorithm == "DuelingDQN":
        custom_objects = {"policy_class": DuelingCnnPolicy}
    model = AlgoClass.load(model_path, env=env, device=device, custom_objects=custom_objects)
    return model


def get_action_meanings(env: VecEnv) -> List[str]:
    """Return action meaning strings from the underlying ALE env."""
    # Walk through wrappers to reach the base environment
    base = env.envs[0]
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    if hasattr(base, "get_action_meanings"):
        return list(base.get_action_meanings())
    # Fallback for Breakout
    return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]


def frames_identical(obs_a: np.ndarray, obs_b: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Return True when two stacked observations are effectively identical.
    Used as a proxy for "game is frozen waiting for FIRE".
    """
    if obs_a is None or obs_b is None:
        return False
    diff = np.abs(obs_a.astype(np.float32) - obs_b.astype(np.float32))
    return float(diff.max()) < tol


def diagnose(
    model_path: str,
    algorithm: str,
    env_id: str = "ALE/Breakout-v5",
    n_episodes: int = 10,
    max_steps_per_episode: int = 5000,
    deterministic: bool = True,
    seed: int = 42,
) -> None:
    print("=" * 70)
    print("FIRE-action diagnostic")
    print("=" * 70)
    print(f"Model      : {model_path}")
    print(f"Algorithm  : {algorithm}")
    print(f"Environment: {env_id}")
    print(f"Episodes   : {n_episodes}")
    print(f"Deterministic policy: {deterministic}")
    print()

    # Training-style env (EpisodicLifeEnv active) so we observe life-loss states
    env = make_atari_env(env_id=env_id, n_envs=1, seed=seed, frame_stack=4)
    model = load_model(model_path, algorithm, env=env)

    meanings = get_action_meanings(env)
    print(f"Action meanings: {meanings}")
    fire_indices = [i for i, m in enumerate(meanings) if "FIRE" in m.upper()]
    print(f"Actions that contain FIRE: {fire_indices} → {[meanings[i] for i in fire_indices]}")
    print()

    # Counters
    waiting_actions = Counter()          # actions chosen while frames are frozen
    all_actions = Counter()              # all actions for context
    waiting_steps = 0
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    lives_lost = 0

    for ep in range(n_episodes):
        obs = env.reset()
        prev_obs = None
        done = False
        ep_reward = 0.0
        ep_len = 0
        steps_waiting_this_ep = 0

        while not done and ep_len < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action[0]) if hasattr(action, "__len__") else int(action)

            all_actions[action] += 1
            total_steps += 1

            # Detect frozen frame (ball on paddle / waiting for launch)
            is_waiting = frames_identical(prev_obs, obs) if prev_obs is not None else False
            if is_waiting:
                waiting_actions[action] += 1
                waiting_steps += 1
                steps_waiting_this_ep += 1

            prev_obs = obs.copy()
            obs, reward, dones, infos = env.step([action])
            done = bool(dones[0])
            ep_reward += float(reward[0])
            ep_len += 1

            # Count life losses via info (EpisodicLifeEnv / ALE)
            info = infos[0] if infos else {}
            if info.get("lives", None) is not None and "ale.lives" in str(info).lower():
                pass  # optional; not required for the main diagnosis

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        print(
            f"  Episode {ep+1:2d}: reward={ep_reward:6.1f}  length={ep_len:4d}  "
            f"waiting_frames={steps_waiting_this_ep:3d}"
        )

    env.close()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total environment steps     : {total_steps}")
    print(f"Steps with frozen frames    : {waiting_steps}  "
          f"({100.0 * waiting_steps / max(total_steps, 1):.1f}% of all steps)")
    print(f"Mean episode reward (clipped): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length         : {np.mean(episode_lengths):.1f}")
    print()

    print("Action distribution — ALL steps")
    print("-" * 40)
    for a in range(len(meanings)):
        cnt = all_actions[a]
        pct = 100.0 * cnt / max(total_steps, 1)
        flag = "  ← FIRE" if a in fire_indices else ""
        print(f"  {a}: {meanings[a]:12s}  {cnt:6d}  ({pct:5.1f}%){flag}")
    print()

    if waiting_steps == 0:
        print("WARNING: No frozen frames were detected.")
        print("         This can happen if the agent never reaches a life-loss")
        print("         state or if the observation changes for other reasons.")
        print("         Try increasing --n_episodes or inspect a video.")
        return

    print("Action distribution — WAITING / FROZEN frames only")
    print("-" * 40)
    fire_count = 0
    for a in range(len(meanings)):
        cnt = waiting_actions[a]
        pct = 100.0 * cnt / waiting_steps
        is_fire = a in fire_indices
        if is_fire:
            fire_count += cnt
        flag = "  ← FIRE" if is_fire else ""
        print(f"  {a}: {meanings[a]:12s}  {cnt:6d}  ({pct:5.1f}%){flag}")
    print()

    fire_pct = 100.0 * fire_count / waiting_steps
    print(f"FIRE (any variant) selected on {fire_count}/{waiting_steps} "
          f"waiting frames  →  {fire_pct:.1f}%")
    print()

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if fire_pct < 20.0:
        print("CONFIRMED: The agent rarely presses FIRE when the ball is")
        print("           sitting on the paddle.  This explains the extremely")
        print("           short episode lengths and low clipped rewards.")
        print()
        print("Recommended actions:")
        print("  • Inspect a few trajectories visually (record a short video).")
        print("  • Consider a temporary helper wrapper that forces FIRE for the")
        print("    first 1–2 steps after every life loss during early training.")
        print("  • Or increase exploration / curriculum so the agent discovers")
        print("    the FIRE action more often in the waiting state.")
    elif fire_pct < 60.0:
        print("PARTIAL: The agent sometimes presses FIRE in the waiting state,")
        print("         but not consistently.  This is likely still limiting")
        print("         performance.")
    else:
        print("OK: The agent selects FIRE frequently when frames are frozen.")
        print("    The short episode lengths / low reward must have another cause")
        print("    (e.g. poor paddle tracking, value-function issues, etc.).")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose FIRE action usage after life loss")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the .zip checkpoint")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["DQN", "DoubleDQN", "DuelingDQN"])
    parser.add_argument("--env_id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=5000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    deterministic = not args.stochastic
    diagnose(
        model_path=args.model_path,
        algorithm=args.algorithm,
        env_id=args.env_id,
        n_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        deterministic=deterministic,
        seed=args.seed,
    )
