"""
train.py — Main Training Entry-Point
======================================
Performs a GPU sanity check, loads hyperparameters from config.yaml (with
optional command-line overrides), builds the environment and agent, wires up
callbacks, and executes the SB3 .learn() loop.

Usage examples
--------------
# Train standard DQN with default config:
    python train.py

# Override algorithm and environment from the command line:
    python train.py --algorithm DoubleDQN --env_id ALE/Pong-v5

# Resume a previous run from the latest checkpoint. The run's own saved
# checkpoints/<env>/<algo>/<run>/config.yaml is loaded automatically instead
# of the top-level config.yaml, so the resumed run keeps the exact
# hyperparameters it started with (CLI flags still override on top of it):
    python train.py --resume checkpoints/ALE-Breakout-v5/DQN/001/DQN_step_500000_steps.zip

# Quick smoke-test (2 M steps, small buffer):
    python train.py --total_timesteps 2000000 --buffer_size 50000
"""

from __future__ import annotations

import argparse
import datetime
import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from agent import create_agent
from environment import make_atari_env, make_eval_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

# =============================================================================
# GPU / CUDA Validation
# =============================================================================


def check_cuda() -> torch.device:
    """
    Verify that a CUDA-capable GPU is available and print hardware info.

    Returns
    -------
    torch.device
        ``device("cuda")`` on success.

    Raises
    ------
    RuntimeError
        Immediately terminates training if CUDA is not available.  A GPU is
        required; CPU training on Atari at 10 M steps is impractical.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "  CUDA IS NOT AVAILABLE — Training aborted.\n\n"
            "  This project requires a CUDA-capable NVIDIA GPU.\n"
            "  Please verify:\n"
            "    1. An NVIDIA GPU is installed and recognised by the OS.\n"
            "    2. The latest NVIDIA driver is installed.\n"
            "    3. PyTorch was installed with CUDA support, e.g.:\n"
            "         pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "    4. Run `nvidia-smi` to confirm the driver is active.\n" + "=" * 60
        )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3

    # Observation tensors are always the same fixed shape
    # (batch_size, frame_stack, 84, 84), so cuDNN's autotuner can safely
    # benchmark and cache the fastest convolution algorithms instead of
    # picking a generic one on every forward pass — a easy speed win on
    # small GPUs like the GTX 1650.
    torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 60)
    print("  CUDA Sanity Check — PASSED")
    print(f"  Device Name  : {props.name}")
    print(f"  CUDA Version : {torch.version.cuda}")
    print(f"  Total VRAM   : {vram_gb:.2f} GB")
    print(f"  SM Count     : {props.multi_processor_count}")
    print("  cuDNN Bench  : enabled")
    print("=" * 60 + "\n")

    return torch.device("cuda")


# =============================================================================
# Run-directory helpers
# =============================================================================


def _next_run_number(algo_dir: Path) -> str:
    """
    Return the next zero-padded three-digit run number (e.g. ``'001'``,
    ``'002'``) by scanning *algo_dir* for existing numeric subdirectories.
    Thread-safe enough for sequential use; the directory is created by the
    caller immediately after this returns.
    """
    existing: list[int] = [
        int(p.name)
        for p in (algo_dir.iterdir() if algo_dir.exists() else [])
        if p.is_dir() and p.name.isdigit()
    ]
    return f"{max(existing, default=0) + 1:03d}"


def _write_run_config(run_dir: Path, config: Dict[str, Any], run_num: str) -> None:
    """
    Write a YAML snapshot of *config* to ``<run_dir>/config.yaml`` so each
    run is self-documenting *and* reloadable.

    This replaces the old Markdown-only snapshot (``config.md``): YAML is
    just as human-readable, but — unlike Markdown — it can also be parsed
    straight back into a dict.  ``--resume`` uses this file (see
    ``load_run_config()``) to continue a run with the exact hyperparameters
    it started with, instead of silently picking up whatever currently
    lives in the project's top-level ``config.yaml``.

    Internal/derived keys (prefixed with ``_``, e.g. ``_tb_log_dir``) are
    excluded since they are recomputed from ``run_dir`` on every load and
    would otherwise go stale if the checkpoint tree is ever moved.
    """
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    # Metadata for humans skimming the file; harmless extra keys on reload
    # since merge_config_with_args only ever adds non-None CLI overrides
    # on top, and train() never reads these two back out.
    snapshot["_run_number"] = run_num
    snapshot["_trained_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(snapshot, fh, default_flow_style=False, sort_keys=False)


def load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load the YAML config snapshot saved by ``_write_run_config()`` at the
    start of a run.

    Parameters
    ----------
    run_dir : Path
        The numbered run directory (e.g. ``checkpoints/ALE-Breakout-v5/DQN/001``).

    Returns
    -------
    dict | None
        The saved config, or ``None`` if no ``config.yaml`` snapshot exists
        in *run_dir* (e.g. a legacy run that only wrote ``config.md``, or a
        ``--resume`` path that doesn't point inside a recognised run
        directory). Callers should fall back to the top-level config.yaml
        in that case.
    """
    run_config_path = run_dir / "config.yaml"
    if not run_config_path.is_file():
        return None
    with run_config_path.open("r") as fh:
        saved = yaml.safe_load(fh)
    # Drop the "_run_number"/"_trained_at" bookkeeping keys written by
    # _write_run_config — they're for humans reading the file, not for
    # feeding back into train()/create_agent().
    return {k: v for k, v in saved.items() if not k.startswith("_")}


def _resolve_run_dir_from_checkpoint(resume_path: str) -> Path:
    """
    Return the numbered run directory that contains *resume_path*.

    Mirrors the ``*_best/`` handling in ``evaluate.py``'s
    ``_resolve_video_dir()``: ``EvalCallback`` saves ``best_model.zip``
    inside an ``<algo>_best/`` subfolder, so when *resume_path* lives there
    we step up one level to land on the numbered run directory (where
    ``config.yaml`` and the regular step checkpoints live) rather than the
    ``*_best/`` subfolder itself.
    """
    parent = Path(resume_path).resolve().parent
    if parent.name.endswith("_best"):
        parent = parent.parent
    return parent


# =============================================================================
# Configuration Loading & CLI Override
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (absolute or relative to the working directory).

    Returns
    -------
    dict
        Parsed hyperparameter dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the given path.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'\n"
            "Run train.py from the atari_benchmark/ directory or pass "
            "--config with the correct path."
        )
    with path.open("r") as fh:
        config = yaml.safe_load(fh)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Every argument is *optional*; when provided it overrides the corresponding
    key in config.yaml.  This lets users run ablations without editing the file:

        python train.py --algorithm DoubleDQN --batch_size 64

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train DQN / DoubleDQN / DuelingDQN on Atari.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Config path --------------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML hyperparameter file.",
    )

    # --- Core overrides -----------------------------------------------------
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["DQN", "DoubleDQN", "DuelingDQN"],
        help="Algorithm variant to train.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        help="Gymnasium/ALE environment ID, e.g. 'ALE/Breakout-v5'.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        help="Total number of environment steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Adam optimiser learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Replay buffer mini-batch size.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        help="Maximum replay buffer capacity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Global random seed.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        help="Number of parallel environments.",
    )

    # --- Checkpoint / resume ------------------------------------------------
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a .zip checkpoint to resume training from. "
            "The timestep counter and replay buffer are restored automatically."
        ),
    )

    # --- Auto-Confirmation for automated runs ------------------------------------------------
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt (for scripted/automated runs).",
    )

    return parser


def merge_config_with_args(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Override config values with non-None CLI arguments.

    Parameters
    ----------
    config : dict
        Base configuration loaded from YAML.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    config.update(overrides)
    return config

def confirm_to_proceed(prompt: str = "Proceed with this configuration? [y/N]: ") -> bool:
    """
    Ask the user to confirm the effective configuration before any
    environment or agent construction begins.

    Accepts 'y' or 'yes' (case-insensitive, surrounding whitespace ignored)
    as confirmation. Anything else -- including empty input, EOF (e.g. when
    stdin isn't a TTY), or Ctrl-C -- cancels the run before check_cuda()'s
    work is followed by any GPU/env allocation.

    Returns
    -------
    bool
        True to proceed, False to cancel.
    """
    try:
        response = input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()  # move past the unfinished prompt line
        return False
    return response.strip().lower() in ("y", "yes")


# =============================================================================
# Callback: per-step progress logging
# =============================================================================


class TrainingProgressCallback(BaseCallback):
    """
    Prints a compact progress line every ``log_every`` timesteps.

    Logs: elapsed wall-clock time, current timestep, FPS, and the most
    recent episode reward (when available from the Monitor wrapper).
    """

    def __init__(self, total_timesteps: int, log_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_every = log_every
        self._start_time: float = 0.0
        self._last_ep_reward: Optional[float] = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Capture latest episode reward if available
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._last_ep_reward = ep["r"]

        # Use a rolling threshold rather than `% log_every == 0` because
        # num_timesteps advances by n_envs per step and won't necessarily
        # land exactly on a multiple of log_every for arbitrary n_envs.
        if self.num_timesteps // self.log_every > getattr(self, "_last_log_bucket", -1):
            self._last_log_bucket = self.num_timesteps // self.log_every
            elapsed = time.time() - self._start_time
            fps = self.num_timesteps / max(elapsed, 1.0)
            progress = 100.0 * self.num_timesteps / self.total_timesteps
            ep_str = (
                f"{self._last_ep_reward:.1f}"
                if self._last_ep_reward is not None
                else "N/A"
            )
            vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
            print(
                f"  [{progress:5.1f}%] step={self.num_timesteps:>9,} | "
                f"fps={fps:>5.0f} | ep_rew={ep_str:>8} | "
                f"elapsed={elapsed / 60:.1f}m | "
                f"VRAM={vram_mb:.0f} MB"
            )
            # Log replay buffer fill % so it appears in TensorBoard
            buf = self.model.replay_buffer
            fill_pct = 100.0 * buf.size() / buf.buffer_size
            self.model.logger.record("train/buffer_fill_pct", fill_pct)
        return True  # returning False would stop training


# =============================================================================
# Callback: config → TensorBoard Text + HParams tabs
# =============================================================================


class ConfigLoggerCallback(BaseCallback):
    """
    Writes the full hyperparameter config into TensorBoard at training
    boundaries so every run is self-documenting:

    * **Text tab**   — human-readable markdown table logged at step 0.
    * **HParams tab** — machine-comparable table logged at training end
                        alongside the run's final episode reward, so you can
                        sort/filter runs by any hyperparameter in TensorBoard.
    """

    def __init__(self, config: Dict[str, Any], verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self._final_ep_rew: float = 0.0

    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        writer = self._tb_writer()
        if writer is None:
            return
        lines = ["| Hyperparameter | Value |", "|:---|:---|"]
        for k, v in self.config.items():
            lines.append(f"| `{k}` | `{v}` |")
        writer.add_text("config/hyperparameters", "\n".join(lines), global_step=0)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._final_ep_rew = float(ep["r"])
        return True

    def _on_training_end(self) -> None:
        writer = self._tb_writer()
        if writer is None:
            return
        # Only scalar / string values are valid for the HParams plugin
        hparam_dict = {
            k: (str(v) if not isinstance(v, (int, float, bool)) else v)
            for k, v in self.config.items()
            if k not in ("log_dir", "checkpoint_dir", "video_dir", "_tb_log_dir")
        }
        writer.add_hparams(
            hparam_dict,
            {"hparam/ep_rew_final": self._final_ep_rew},
        )

    # ------------------------------------------------------------------
    def _tb_writer(self):
        """Return the underlying SummaryWriter, or None if not logging to TB."""
        from stable_baselines3.common.logger import TensorBoardOutputFormat

        for fmt in self.model.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                return fmt.writer
        return None


# =============================================================================
# Main Training Routine
# =============================================================================


def train(config: Dict[str, Any], resume_path: Optional[str] = None) -> None:
    """
    Build, optionally restore, and train a DQN-family agent.

    Parameters
    ----------
    config : dict
        Fully merged hyperparameter dictionary.
    resume_path : str | None
        If provided, load the model from this checkpoint and continue training.
    """

    # ---- Paths -------------------------------------------------------------
    env_tag = config["env_id"].replace("/", "-").replace(":", "_")
    algo = config["algorithm"]
    lr = config["learning_rate"]
    bs = config["batch_size"]

    run_label = f"{algo}_lr{lr}_batch{bs}"
    algo_dir = Path(config["checkpoint_dir"]) / env_tag / algo

    # When resuming, reuse the existing numbered run directory so all
    # checkpoints from a resumed run stay together.  For new runs, mint
    # the next available number (001, 002, …).
    #
    # _resolve_run_dir_from_checkpoint() steps past an EvalCallback
    # "<algo>_best/" subfolder (best_model.zip lives one level deeper than
    # the numbered run dir) — using it here (instead of a separate, looser
    # check) keeps this in sync with how --resume's config.yaml lookup
    # resolves the run dir in __main__, so resuming from best_model.zip
    # continues in the original run dir instead of silently minting a new
    # one.
    if resume_path:
        _resume_target = Path(resume_path).resolve()
        if _resume_target.is_dir() and _resume_target.name.isdigit():
            # The run directory itself was passed instead of a checkpoint file.
            _resolved_run_dir = _resume_target
        else:
            _resolved_run_dir = _resolve_run_dir_from_checkpoint(resume_path)

        if _resolved_run_dir.name.isdigit() and _resolved_run_dir.parent == algo_dir.resolve():
            run_dir = _resolved_run_dir
        else:
            # Legacy path without a numbered subfolder — create a new run.
            run_dir = algo_dir / _next_run_number(algo_dir)
    else:
        run_dir = algo_dir / _next_run_number(algo_dir)

    run_num = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep using the name 'checkpoint_dir' for the rest of this function
    # so it reads naturally at every call-site below.
    checkpoint_dir = run_dir

    # ---- Environments ------------------------------------------------------
    print(f"Building training environment: {config['env_id']}")
    train_env = make_atari_env(
        env_id=config["env_id"],
        n_envs=config["n_envs"],
        seed=config["seed"],
        frame_stack=config["frame_stack"],
        reward_clip_mode=config["reward_clip_mode"],
    )

    # Print the wrapper chain so it is obvious whether ForceFireOnLifeLoss
    # (and the rest of the DeepMind pipeline) is present.
    def _wrapper_chain(env) -> str:
        """Return a human-readable list of wrapper class names."""
        names = []
        # VecEnv wrappers (outermost first)
        cur = env
        while cur is not None:
            names.append(type(cur).__name__)
            # SubprocVecEnv / DummyVecEnv expose .envs; other VecEnv wrappers use .venv
            if hasattr(cur, "venv"):
                cur = cur.venv
            elif hasattr(cur, "envs"):
                # Dive into the first sub-environment's Gymnasium wrapper stack
                sub = cur.envs[0]
                while sub is not None:
                    names.append(type(sub).__name__)
                    sub = getattr(sub, "env", None)
                break
            else:
                break
        return " → ".join(names)

    train_chain = _wrapper_chain(train_env)
    print(f"Training env wrapper chain:\n  {train_chain}")
    # NOTE: ForceFireOnLifeLoss is intentionally NEVER part of the training
    # chain (see the NOTE comment in environment.py's make_atari_env) --
    # replacing the policy's chosen action inside step() would corrupt the
    # replay buffer SB3 is simultaneously writing to. Training instead
    # relies on EpisodicLifeEnv's reset-on-life-loss cycle to trigger
    # FireResetEnv naturally. The wrapper belongs only in the eval chain,
    # verified below.

    print("Building evaluation environment …")
    eval_env = make_eval_env(
        env_id=config["env_id"],
        seed=config["seed"] + 100,  # Different seed from training
        frame_stack=config["frame_stack"],
    )
    eval_chain = _wrapper_chain(eval_env)
    print(f"Eval env wrapper chain:\n  {eval_chain}")
    if "FireResetEnv" in eval_chain:
        if "ForceFireOnLifeLoss" in eval_chain:
            print("  ✓ ForceFireOnLifeLoss is active (fire-start game)")
        else:
            print("  ✗ ForceFireOnLifeLoss NOT found on a fire-start game — check environment.py")
    else:
        print("  – Game does not require FIRE-to-start; ForceFireOnLifeLoss correctly not applied")

    # ---- Config snapshot ---------------------------------------------------
    # Write before the agent is built so the file exists even if training
    # crashes during environment / model construction.
    if not resume_path:  # Don't overwrite the original config on resume
        config["_tb_log_dir"] = str(run_dir / "tb_logs")
        print(f"Run {run_num} → {run_dir}")
    else:
        config["_tb_log_dir"] = str(run_dir / "tb_logs")
        print(f"Resuming run {run_num} → {run_dir}")

    # ---- Agent (new or resumed) --------------------------------------------
    if resume_path:
        print(f"\nResuming from checkpoint: {resume_path}")
        from agent import _ALGORITHM_REGISTRY

        AlgorithmClass = _ALGORITHM_REGISTRY[algo]
        agent = AlgorithmClass.load(
            path=resume_path,
            env=train_env,
            device="cuda",
        )
        elapsed_steps = agent.num_timesteps
        remaining_steps = config["total_timesteps"] - elapsed_steps
        reset_timesteps = False
        print(
            f"Restored at step {elapsed_steps:,}. "
            f"Continuing for {remaining_steps:,} more steps."
        )
        # Point agent's TB logger at the run dir
        agent.tensorboard_log = str(run_dir / "tb_logs")
    else:
        agent = create_agent(config, train_env)
        remaining_steps = config["total_timesteps"]
        reset_timesteps = True
        _write_run_config(run_dir, config, run_num)


    # ---- Callbacks ---------------------------------------------------------
    # 1. Checkpoint: save model weights every N steps to guard against crashes.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(config["checkpoint_interval"] // config["n_envs"], 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{algo}_step",
        save_replay_buffer=False,  # Replay buffers can be very large (several GB)
        save_vecnormalize=False,
        verbose=1,
    )

    # 2. Evaluation: run N deterministic episodes every checkpoint_interval steps and log
    #    mean_reward to TensorBoard.  Best model is saved automatically.
    best_model_path = str(checkpoint_dir / f"{algo}_best")
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_path,
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=max(200_000 // config["n_envs"], 1),
        n_eval_episodes=config.get("n_eval_episodes", 15),
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 3. Progress: human-readable console progress every 10 k steps.
    progress_cb = TrainingProgressCallback(
        total_timesteps=config["total_timesteps"],
        log_every=10_000,
    )

    # 4. Config logger: writes hyperparameters to TensorBoard Text + HParams tab.
    config_cb = ConfigLoggerCallback(config=config)

    callback_list = CallbackList([checkpoint_cb, eval_cb, progress_cb, config_cb])

    # ---- Training ----------------------------------------------------------
    vram_alloc = torch.cuda.memory_allocated() / 1024 ** 2
    vram_reserv = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"GPU memory after model load — allocated: {vram_alloc:.0f} MB | reserved: {vram_reserv:.0f} MB")
    print(f"\nStarting training — {remaining_steps:,} steps …\n")
    t0 = time.time()

    agent.learn(
        total_timesteps=remaining_steps,
        callback=callback_list,
        log_interval=4,
        tb_log_name=run_label,
        reset_num_timesteps=reset_timesteps,
        progress_bar=False,  # Using our own progress callback above
    )

    wall_time = (time.time() - t0) / 3600
    print(f"\nTraining complete in {wall_time:.2f} h.")

    # ---- Save final model --------------------------------------------------
    final_path = checkpoint_dir / f"{algo}_final"
    agent.save(str(final_path))
    print(f"Final model saved → {final_path}.zip")

    train_env.close()
    eval_env.close()

    # Explicitly free the replay buffer (can be several GB) before returning
    # so the next algorithm run can allocate its own buffer without OOM.
    del agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # 1. GPU check — hard-fail if no CUDA device is found
    check_cuda()

    # 2. Parse arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    # 3. Load config.
    #    --resume reuses the checkpoint's OWN run directory config.yaml
    #    (written by _write_run_config at the start of that run) instead of
    #    the top-level --config file, so a resumed run keeps the exact
    #    hyperparameters it started with even if config.yaml has since
    #    changed. CLI flags are still applied on top as overrides. Falls
    #    back to --config with a warning for legacy runs that only saved
    #    the old config.md (no machine-readable snapshot to load).
    if args.resume:
        resume_run_dir = _resolve_run_dir_from_checkpoint(args.resume)
        run_config = load_run_config(resume_run_dir)
        if run_config is not None:
            print(f"Resuming: loaded saved config from {resume_run_dir / 'config.yaml'}")
            config = run_config
        else:
            print(
                f"WARNING: no config.yaml snapshot found in '{resume_run_dir}' "
                f"(legacy run?). Falling back to '{args.config}' — "
                "hyperparameters may not match the original run."
            )
            config = load_config(args.config)
    else:
        config = load_config(args.config)

    # 3b. Merge CLI overrides on top of whichever config was loaded above.
    config = merge_config_with_args(config, args)

    print("\nEffective configuration:")
    for k, v in config.items():
        print(f"  {k:<30} {v}")
    print()

    # 4. Determine which algorithms to train.
    #    --algorithm trains only that one; omitting it trains all three for
    #    a fresh run. On --resume there is exactly one checkpoint to resume,
    #    so default to the algorithm recorded in the resumed run's config
    #    instead of re-training all three from scratch.
    if args.resume:
        algorithms = [args.algorithm] if args.algorithm is not None else [config["algorithm"]]
    else:
        algorithms = (
            [args.algorithm]
            if args.algorithm is not None
            else ["DQN", "DoubleDQN", "DuelingDQN"]
        )

    print(f"Algorithms to train: {', '.join(algorithms)}\n")

    # 4b. Confirm before any environment/agent/CUDA allocation happens
    #     inside train(). Placed after every detail (config + algorithm
    #     list) is printed so there's nothing left to check.
    if not args.yes and not confirm_to_proceed():
        print("Aborted by user — no environment or agent was created.")
        raise SystemExit(0)

    # 5. Train each algorithm in sequence with the same config.
    #TODO: profiler just in case
    for i, algo in enumerate(algorithms, 1):
        print(f"\n{'#' * 60}")
        print(f"#  [{i}/{len(algorithms)}] Starting: {algo}")
        print(f"{'#' * 60}\n")
        algo_config = {**config, "algorithm": algo}
        train(algo_config, resume_path=args.resume)
    # import cProfile
    # import pstats

    # for i, algo in enumerate(algorithms, 1):
    #     print(f"\n{'#' * 60}")
    #     print(f"#  [{i}/{len(algorithms)}] Starting: {algo}")
    #     print(f"{'#' * 60}\n")
    #     algo_config = {**config, "algorithm": algo}

    #     profiler = cProfile.Profile()
    #     profiler.enable()
    #     try:
    #         train(algo_config, resume_path=args.resume)
    #     finally:
    #         profiler.disable()
    #         profiler.dump_stats("profile.out")
    #         pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)

        # Force memory release between runs on memory-constrained systems.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  All {len(algorithms)} algorithm(s) finished.")
    print(f"  Checkpoints : {config['checkpoint_dir']}/")
    print(f"  TensorBoard : tensorboard --logdir {config['log_dir']}/")
    print(f"{'=' * 60}\n")