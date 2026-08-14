"""
environment.py — Atari Environment Factory
==========================================
Constructs and wraps Atari environments following the DeepMind preprocessing
pipeline described in Mnih et al. (2015) "Human-level control through deep
reinforcement learning."

Preprocessing stages applied (in order):
  1. NoopResetEnv      — Sample k ~ Uniform[1, noop_max] no-op actions on reset
                          to introduce stochasticity in starting positions.
  2. MaxAndSkipEnv     — Repeat each action for `skip` frames; return max-pooled
                          observation over the last 2 frames to remove flicker.
  3. EpisodicLifeEnv   — Signal episode termination on every life loss during
                          training (helps value estimation).  Disabled at eval.
  4. FireResetEnv      — Automatically press FIRE on reset for games that require
                          it to start (e.g. Breakout).
  4b.ForceFireOnLifeLoss — Force FIRE for a short number of steps after every
                          life loss (and on reset).  Training bootstrap only;
                          ensures the ball is launched so the agent can learn
                          from real rallies.  Disabled at eval.
  5. WarpFrame         — Convert to grayscale and resize to 84 × 84 pixels.
  6. ClipRewardEnv     — Clip rewards to {-1, 0, +1} via np.sign for training
                          stability across different score scales. Disabled at eval.
  7. VecFrameStack     — Stack 4 consecutive processed frames as one observation,
                          giving the agent temporal information.
  8. VecTransposeImage — Reorder axes from (H, W, C) to (C, H, W) so that
                          PyTorch's Conv2d layers receive channel-first tensors.

Two public functions are exported:
  make_atari_env()   — Training environment (EpisodicLife + ClipReward enabled).
  make_eval_env()    — Evaluation environment (raw scores, full lives).
"""

from __future__ import annotations

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)  # register ALE/Atari environments


# ---------------------------------------------------------------------------
# Fire-reset detection
# ---------------------------------------------------------------------------


def _detect_fire_needed(env: gym.Env) -> bool:
    """
    Return True only if FIRE must be pressed after reset to begin play.

    Strategy: compare the raw pixel observation immediately after reset
    with the observation after a single NOOP step.

    * If they are **identical**, the ROM is frozen — no game element moved
      without player input.  Since FIRE is in the action set we conclude
      it is the start trigger (e.g. Breakout holds the ball on the paddle
      until FIRE is pressed).

    * If the observation **changes** on NOOP the game is already advancing
      on its own (e.g. Pong auto-launches the ball after reset), so
      FireResetEnv is not needed and should be skipped to avoid pressing
      an unintended FIRE action at each episode boundary.

    The detection is run on the unwrapped base environment before any
    preprocessing wrappers are applied, so it sees the raw RGB pixels
    and is not affected by grayscale conversion or frame stacking.
    """
    if "FIRE" not in env.unwrapped.get_action_meanings():  # type: ignore[union-attr]
        return False
    obs_reset, _ = env.reset()
    obs_noop, _, terminated, truncated, _ = env.step(0)  # one NOOP step
    env.reset()  # restore clean state before any wrappers are applied
    if terminated or truncated:
        # Immediate termination on NOOP — uncommon, but treat as needs-FIRE.
        return True
    return bool(np.array_equal(obs_reset, obs_noop))


from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


# ---------------------------------------------------------------------------
# Force FIRE after life loss (training bootstrap)
# ---------------------------------------------------------------------------


class ForceFireOnLifeLoss(gym.Wrapper):
    """
    Force the FIRE action for a short number of steps after every life loss
    and after a true environment reset.

    Rationale
    ---------
    ``FireResetEnv`` only injects FIRE on a full ``reset()``.  After an
    intermediate life loss handled by ``EpisodicLifeEnv`` the ball sits on
    the paddle and the agent must press FIRE itself.  Many agents fail to
    discover this action, producing extremely short episodes.  This wrapper
    guarantees the ball is launched so the agent can experience real
    rallies and learn paddle control.

    The wrapper is intended for **training only**.  Evaluation environments
    should remain free of forced actions.
    """

    def __init__(self, env: gym.Env, force_steps: int = 1):
        super().__init__(env)
        self.force_steps = max(int(force_steps), 0)
        self._remaining = 0
        self._prev_lives: int | None = None

        # Resolve the FIRE action index from the underlying ALE action set.
        meanings = env.unwrapped.get_action_meanings()  # type: ignore[union-attr]
        if "FIRE" in meanings:
            self.fire_action = meanings.index("FIRE")
        else:
            # Fallback for the minimal Breakout action set.
            self.fire_action = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._remaining = self.force_steps
        # Initialise life counter from info if available.
        lives = info.get("lives", info.get("ale.lives"))
        self._prev_lives = int(lives) if lives is not None else None
        return obs, info

    def step(self, action):
        if self._remaining > 0:
            action = self.fire_action
            self._remaining -= 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Detect intermediate life loss (lives decreased but game not over).
        lives = info.get("lives", info.get("ale.lives"))
        if lives is not None:
            lives = int(lives)
            if self._prev_lives is not None and lives < self._prev_lives and lives > 0:
                self._remaining = self.force_steps
            self._prev_lives = lives

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Training environment
# ---------------------------------------------------------------------------


def make_atari_env(
    env_id: str,
    n_envs: int = 1,
    seed: int = 42,
    frame_stack: int = 4,
) -> VecTransposeImage:
    """
    Build a vectorised, frame-stacked Atari training environment.

    EpisodicLife and reward clipping are enabled to match the training
    conditions used by Mnih et al. (2015) and subsequent Atari benchmarks.

    Parameters
    ----------
    env_id : str
        Gymnasium/ALE environment identifier, e.g. ``"ALE/Breakout-v5"``.
    n_envs : int
        Number of independent parallel environments.  Values > 1 are run
        with ``SubprocVecEnv`` (one OS process per environment) so that
        frame collection is parallelised across CPU cores.  This keeps the
        GPU fed with fresh transitions instead of sitting idle while a
        single environment steps, without adding to GPU VRAM usage — env
        stepping happens entirely on CPU.
    seed : int
        Base random seed.  Sub-environment i receives seed ``seed + i``.
    frame_stack : int
        Number of consecutive frames to stack into one observation tensor.

    Returns
    -------
    VecTransposeImage
        A fully wrapped, channel-first VecEnv ready for SB3 training.

    Observation shape
    -----------------
    ``(frame_stack, 84, 84)``  —  e.g. ``(4, 84, 84)`` with ``frame_stack=4``.
    dtype: ``uint8``, values in ``[0, 255]``.

    Notes
    -----
    ``FireResetEnv`` is applied **only** when the game genuinely requires a
    FIRE press to begin play (e.g. Breakout holds the ball on the paddle
    until FIRE is pressed).  For games where FIRE is purely a gameplay
    action — jumping in Donkey Kong, or games that auto-launch like Pong —
    the wrapper is skipped to avoid forcing an unintended action at every
    episode / life-loss boundary.
    """

    # Probe a temporary environment once to determine whether FireResetEnv
    # should be applied.  Running this outside the per-env closure means the
    # check executes only once regardless of n_envs.
    _probe = gym.make(env_id, render_mode="rgb_array")
    fire_on_reset: bool = _detect_fire_needed(_probe)
    _probe.close()

    def _make_single_env(rank: int):
        """Factory closure for a single sub-environment."""

        def _init() -> gym.Env:
            env = gym.make(env_id, render_mode="rgb_array")

            # Stage 1 — random no-ops at episode start
            env = NoopResetEnv(env, noop_max=30)
            # Stage 2 — frame skip with max-pooling (reduces 60 fps → ~15 fps)
            env = MaxAndSkipEnv(env, skip=4)
            # Stage 3 — episodic life (training only)
            env = EpisodicLifeEnv(env)
            # Stage 4 — press FIRE only for games that require it to start
            # (detected via _detect_fire_needed; skipped for games like Pong
            # where the ball auto-launches, or DK where FIRE means jump).
            if fire_on_reset:
                env = FireResetEnv(env)
            # Stage 4b — force FIRE after every life loss (and on reset).
            # Guarantees the ball is launched so the agent can learn from
            # real rallies.  Training only; evaluation is left unchanged.
            if fire_on_reset:
                env = ForceFireOnLifeLoss(env, force_steps=1)
            # Stage 5 — grayscale + resize to 84 × 84
            env = WarpFrame(env)
            # Stage 6 — clip rewards to {-1, 0, +1}
            env = ClipRewardEnv(env)
            # Monitor tracks episodic reward and length for TensorBoard
            env = Monitor(env)
            # Seed the fully-wrapped env so NoopResetEnv's RNG is seeded
            # correctly. Must come AFTER all wrappers so the seed propagates
            # through the wrapper chain to the underlying gymnasium env.
            env.reset(seed=seed + rank)
            return env

        return _init

    env_fns = [_make_single_env(rank=i) for i in range(n_envs)]
    # SubprocVecEnv runs each environment in its own OS process so that
    # frame collection (CPU-bound ALE stepping + preprocessing) happens in
    # parallel across cores.  DummyVecEnv would step all n_envs sequentially
    # in the main process, leaving the GPU starved of new data between
    # gradient updates.  Fall back to DummyVecEnv for n_envs=1 to avoid the
    # unnecessary process-spawn overhead.
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    # Stage 7 — stack 4 frames along the channel axis: (H, W, 1) → (H, W, 4)
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    # Stage 8 — channel-first for PyTorch: (H, W, C) → (C, H, W)
    vec_env = VecTransposeImage(vec_env)

    return vec_env


# ---------------------------------------------------------------------------
# Evaluation environment
# ---------------------------------------------------------------------------


def make_eval_env(
    env_id: str,
    seed: int = 0,
    frame_stack: int = 4,
    render_mode: str = "rgb_array",
) -> VecTransposeImage:
    """
    Build a single-env evaluation environment with no training-specific hacks.

    Key differences from the training env:
    * **No** ``EpisodicLifeEnv`` — the agent plays through all lives so that
      episode rewards reflect true game scores.
    * **No** ``ClipRewardEnv`` — raw reward values are returned for fair
      comparison of absolute performance across algorithms.

    Parameters
    ----------
    env_id : str
        Gymnasium/ALE environment identifier.
    seed : int
        Random seed for the evaluation run.
    frame_stack : int
        Number of consecutive frames to stack (must match training setting).
    render_mode : str
        ``"rgb_array"`` for video recording (default), ``"human"`` for live
        on-screen display.

    Returns
    -------
    VecTransposeImage
        A single wrapped evaluation environment.
    """

    # Use the same fire-reset detection as the training env for consistency.
    _probe = gym.make(env_id, render_mode="rgb_array")
    fire_on_reset: bool = _detect_fire_needed(_probe)
    _probe.close()

    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # NOTE: EpisodicLifeEnv intentionally omitted — full lives for scoring.
        if fire_on_reset:
            env = FireResetEnv(env)
        env = WarpFrame(env)
        # NOTE: ClipRewardEnv intentionally omitted — raw scores for reporting.
        env = Monitor(env)
        # Seed after all wrappers so the RNG state is set on the full chain.
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    vec_env = VecTransposeImage(vec_env)

    return vec_env
