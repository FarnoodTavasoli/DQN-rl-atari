"""
evaluate.py — Evaluation & Video Recording Script
===================================================
Loads a trained model snapshot, runs it for N episodes in an evaluation
environment (no EpisodicLife, no reward clipping), and prints summary stats.

Optionally records an MP4 video of the first evaluation episode using
SB3's VecVideoRecorder.

Usage examples
--------------
# Evaluate a saved DQN model (5 episodes, print stats):
    python evaluate.py \\
        --model_path checkpoints/ALE-Breakout-v5/DQN/DQN_final.zip \\
        --algorithm DQN \\
        --env_id ALE/Breakout-v5

# Evaluate DoubleDQN and record a video:
    python evaluate.py \\
        --model_path checkpoints/ALE-Breakout-v5/DoubleDQN/DoubleDQN_final.zip \\
        --algorithm DoubleDQN \\
        --env_id ALE/Breakout-v5 \\
        --record_video \\
        --video_dir videos/DoubleDQN

# Compare all three algorithms by calling this script three times and piping
# stdout to a text file:
    for algo in DQN DoubleDQN DuelingDQN; do
        python evaluate.py --algorithm $algo \\
            --model_path checkpoints/ALE-Breakout-v5/$algo/${algo}_final.zip \\
            --env_id ALE/Breakout-v5 >> results.txt
    done
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from agent import DoubleDQN, DuelingCnnPolicy, DuelingDQN, StandardDQN
from environment import make_eval_env
from stable_baselines3.common.vec_env import VecVideoRecorder

# =============================================================================
# Algorithm registry
# =============================================================================

# Maps CLI name → SB3-compatible class so .load() uses the correct train()
_ALGORITHM_REGISTRY = {
    "DQN": StandardDQN,
    "DoubleDQN": DoubleDQN,
    "DuelingDQN": DuelingDQN,
}


# =============================================================================
# Model loading
# =============================================================================


def load_model(
    model_path: str,
    algorithm: str,
    env=None,
    device: str = "cuda",
):
    """
    Load a trained SB3 model from a .zip snapshot.

    DuelingDQN requires ``DuelingCnnPolicy`` to be registered as a
    ``custom_objects`` entry so that SB3's pickle-based deserialiser can
    locate the class during loading.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the ``*.zip`` checkpoint.
    algorithm : str
        One of ``"DQN"``, ``"DoubleDQN"``, ``"DuelingDQN"``.
    env : VecEnv | None
        Optional environment to bind the model to (needed for further training;
        can be ``None`` for pure evaluation).
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    DQN
        Loaded model with weights on the target device.

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    ValueError
        If the algorithm name is not in the registry.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: '{model_path}'\n"
            "Double-check the path or run train.py first."
        )
    if algorithm not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {list(_ALGORITHM_REGISTRY.keys())}"
        )

    ModelClass = _ALGORITHM_REGISTRY[algorithm]

    # DuelingCnnPolicy is a custom class; pass it as a custom_object so
    # SB3's deserialiser can find it even if it wasn't in the original
    # module search path during saving.
    custom_objects = {}
    if algorithm == "DuelingDQN":
        custom_objects["policy_class"] = DuelingCnnPolicy

    model = ModelClass.load(
        path=str(path),
        env=env,
        device=device,
        custom_objects=custom_objects if custom_objects else None,
    )

    print(f"Loaded {algorithm} model from: {path}")
    print(f"  Timesteps trained : {model.num_timesteps:,}")
    print(f"  Device            : {model.device}")
    return model


# =============================================================================
# Core evaluation loop
# =============================================================================


def run_evaluation(
    model,
    env,
    n_episodes: int,
) -> Tuple[List[float], List[int]]:
    """
    Run the model deterministically for ``n_episodes`` full episodes.

    Episodes are tracked via the Monitor wrapper embedded in the eval env.
    The VecEnv auto-resets at episode boundaries, so we collect the
    terminal episode statistics from the ``"episode"`` key in ``info``.

    Parameters
    ----------
    model : DQN
        Loaded SB3 model with a .predict() method.
    env : VecEnv
        Evaluation environment (n_envs=1, no EpisodicLife, no reward clip).
    n_episodes : int
        Number of complete episodes to run.

    Returns
    -------
    episode_rewards : list[float]
        Raw (unclipped) total reward for each episode.
    episode_lengths : list[int]
        Total number of environment steps for each episode.
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    obs = env.reset()

    while len(episode_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, info = env.step(action)

        # The Monitor wrapper injects "episode" into info at episode end
        if done[0]:
            ep_info = info[0].get("episode")
            if ep_info is not None:
                episode_rewards.append(float(ep_info["r"]))
                episode_lengths.append(int(ep_info["l"]))
            # If Monitor is absent (shouldn't happen), fall back to raw reward
            # (this path is a safety net and will produce inaccurate totals)

    return episode_rewards, episode_lengths


# =============================================================================
# Video recording helpers
# =============================================================================


def wrap_with_video_recorder(
    env,
    video_dir: str,
    algorithm: str,
    env_id: str,
    model_path: str,
    video_length: int = 18_000,
) -> VecVideoRecorder:
    """
    Wrap the eval env with SB3's VecVideoRecorder to capture an MP4.

    The recording starts immediately at step 0 and runs for at most
    ``video_length`` steps.  For Atari with 4-frame skip, 18 000 steps
    corresponds to roughly 5 minutes of gameplay.

    Parameters
    ----------
    env : VecEnv
        The evaluation environment to wrap.
    video_dir : str
        Directory where the ``*.mp4`` file will be written.
    algorithm : str
        Used for the video filename prefix.
    env_id : str
        Used for the video filename prefix.
    model_path : str
        Path to the loaded checkpoint; its stem is appended to the filename.
    video_length : int
        Maximum number of steps to record.

    Returns
    -------
    VecVideoRecorder
        Wrapped environment that automatically saves the video on ``close()``.
    """
    os.makedirs(video_dir, exist_ok=True)
    safe_env_id = env_id.replace("/", "-").replace(":", "_")
    model_stem = Path(model_path).stem
    name_prefix = f"{algorithm}_{safe_env_id}_{model_stem}"

    recorder = VecVideoRecorder(
        venv=env,
        video_folder=video_dir,
        record_video_trigger=lambda step: step == 0,  # Start at the very first step
        video_length=video_length,
        name_prefix=name_prefix,
    )
    print(f"Recording video → {video_dir}/{name_prefix}-*.mp4")
    return recorder


# =============================================================================
# Results reporting
# =============================================================================


def print_results(
    algorithm: str,
    env_id: str,
    episode_rewards: List[float],
    episode_lengths: List[int],
    model_path: str,
) -> None:
    """
    Print a formatted evaluation summary to stdout.

    Parameters
    ----------
    algorithm : str
        Name of the algorithm being evaluated.
    env_id : str
        Environment identifier.
    episode_rewards : list[float]
        Per-episode total reward.
    episode_lengths : list[int]
        Per-episode step count.
    model_path : str
        Source checkpoint path (for reference).
    """
    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    min_r = np.min(episode_rewards)
    max_r = np.max(episode_rewards)
    mean_l = np.mean(episode_lengths)

    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  Evaluation Results")
    print(separator)
    print(f"  Algorithm      : {algorithm}")
    print(f"  Environment    : {env_id}")
    print(f"  Model          : {model_path}")
    print(f"  Episodes run   : {len(episode_rewards)}")
    print(separator)
    print(f"  Mean Score     : {mean_r:>10.2f}")
    print(f"  Std  Score     : {std_r:>10.2f}")
    print(f"  Min  Score     : {min_r:>10.2f}")
    print(f"  Max  Score     : {max_r:>10.2f}")
    print(f"  Mean Length    : {mean_l:>10.0f} steps")
    print(separator)
    print("  Per-episode breakdown:")
    for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths), 1):
        print(f"    Episode {i:>2d}: reward={r:>8.1f}  length={l:>6d}")
    print(f"{separator}\n")


# =============================================================================
# Entry point
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for evaluate.py."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN-family model on Atari.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the .zip checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["DQN", "DoubleDQN", "DuelingDQN"],
        help="Algorithm variant used during training.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="ALE/Breakout-v5",
        help="Gymnasium/ALE environment ID.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="Number of complete evaluation episodes.",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=4,
        help="Number of stacked frames (must match training setting).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the evaluation environment.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="If set, save an MP4 video of the first evaluation episode.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Directory where the recorded MP4 will be saved.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=18_000,
        help="Maximum number of steps to record in the video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    return parser


def main() -> None:
    """Parse arguments, load model, evaluate, and optionally record video."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # ---- Device check (warn but don't abort — inference can run on CPU) ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: CUDA requested but not available. "
            "Falling back to CPU for evaluation."
        )
        args.device = "cpu"
    else:
        print(f"Using device: {args.device}")
        if args.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build eval environment -------------------------------------------
    print(f"\nBuilding evaluation environment: {args.env_id}")
    env = make_eval_env(
        env_id=args.env_id,
        seed=args.seed,
        frame_stack=args.frame_stack,
        render_mode="rgb_array",
    )

    # ---- Optionally wrap with video recorder ------------------------------
    if args.record_video:
        env = wrap_with_video_recorder(
            env=env,
            video_dir=args.video_dir,
            algorithm=args.algorithm,
            env_id=args.env_id,
            model_path=args.model_path,
            video_length=args.video_length,
        )

    # ---- Load model --------------------------------------------------------
    model = load_model(
        model_path=args.model_path,
        algorithm=args.algorithm,
        env=env,
        device=args.device,
    )

    # ---- Run evaluation ----------------------------------------------------
    print(f"\nRunning {args.n_episodes} evaluation episodes …")
    episode_rewards, episode_lengths = run_evaluation(
        model=model,
        env=env,
        n_episodes=args.n_episodes,
    )

    # ---- Close env (flushes video if recording) ----------------------------
    env.close()

    # ---- Print results -----------------------------------------------------
    print_results(
        algorithm=args.algorithm,
        env_id=args.env_id,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        model_path=args.model_path,
    )

    # Return values for programmatic use (e.g. from a sweep script)
    return np.mean(episode_rewards), np.std(episode_rewards)


if __name__ == "__main__":
    main()
