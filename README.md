# Atari DQN Benchmark

A modular, GPU-accelerated benchmark study comparing three Q-learning algorithms on Atari games:

| Algorithm | Reference | Key difference |
|---|---|---|
| **DQN** | Mnih et al. (2015) | Target network selects *and* evaluates the greedy action |
| **Double DQN** | van Hasselt et al. (2016) | Online network selects; target network evaluates (reduces overestimation bias) |
| **Dueling DQN** | Wang et al. (2016) + Double Q-learning | Separate value V(s) and advantage A(s,a) streams in the network head |

---

## Project Layout

```
atari_benchmark/
├── config.yaml        ← Central hyperparameter file (edit here, not in code)
├── environment.py     ← Atari environment factory with DeepMind preprocessing
├── agent.py           ← StandardDQN / DoubleDQN / DuelingDQN implementations
├── train.py           ← Main training script (CUDA check, callbacks, .learn())
├── evaluate.py        ← Evaluation + optional MP4 recording
├── requirements.txt   ← Python dependency list
└── README.md          ← This file
```

Generated at runtime:
```
atari_benchmark/
├── checkpoints/       ← Model snapshots saved every 100k steps
├── logs/              ← TensorBoard event files
└── videos/            ← Evaluation MP4 recordings (optional)
```

---

## Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **GPU**: NVIDIA GPU with CUDA support (training is GPU-mandatory by design)
- **Python**: 3.9 – 3.12
- **CUDA**: 11.8 or newer (check with `nvidia-smi`)

---

## Installation

### Step 1 — Create and activate a virtual environment

```bash
# Using conda (recommended)
conda create -n atari-dqn python=3.11 -y
conda activate atari-dqn

# Or using venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### Step 2 — Install PyTorch with CUDA

Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 3 — Install project dependencies

```bash
cd atari_benchmark/
pip install -r requirements.txt
```

### Step 4 — Accept the Atari ROMs licence

```bash
pip install gymnasium[accept-rom-license]
python -m ale_py.roms   # confirms ROMs are installed
```

---

## Configuration

All hyperparameters live in `config.yaml`. The most important settings:

```yaml
env_id: "ALE/Breakout-v5"    # Change to any ALE environment
algorithm: "DQN"              # DQN | DoubleDQN | DuelingDQN
learning_rate: 2.5e-4
batch_size: 32
buffer_size: 100000           # Raise to 1000000 if you have ≥16 GB RAM
total_timesteps: 10000000     # 10M steps ≈ 4–12 h on a mid-range GPU
```

Every key in `config.yaml` can be overridden on the command line — no need to edit the file for quick ablations.

---

## Training

### Basic usage (reads config.yaml)

```bash
python train.py
```

### Override algorithm and environment

```bash
python train.py --algorithm DoubleDQN --env_id ALE/Pong-v5
```

### Full example with common overrides

```bash
python train.py \
    --algorithm DuelingDQN \
    --env_id ALE/SpaceInvaders-v5 \
    --total_timesteps 5000000 \
    --learning_rate 1e-4 \
    --batch_size 64
```

### Resume after a crash or thermal throttle

```bash
python train.py \
    --algorithm DQN \
    --resume checkpoints/ALE-Breakout-v5/DQN/DQN_step_3000000_steps.zip
```

The `--resume` flag restores the timestep counter and continues for the remaining steps.

### Checkpointing

Model weights are saved every **100,000 steps** (configurable via `checkpoint_interval`) to:
```
checkpoints/<env>/<algorithm>/
  <algorithm>_step_100000_steps.zip
  <algorithm>_step_200000_steps.zip
  ...
  <algorithm>_best/           ← best checkpoint by eval reward
  <algorithm>_final.zip       ← saved at end of training
```

---

## Monitoring with TensorBoard

```bash
# From the atari_benchmark/ directory:
tensorboard --logdir logs/

# Then open http://localhost:6006 in your browser
```

The log directory is structured as:
```
logs/
└── ALE-Breakout-v5/
    ├── DQN_lr2.5e-04_batch32_1/
    ├── DoubleDQN_lr2.5e-04_batch32_1/
    └── DuelingDQN_lr2.5e-04_batch32_1/
```

Key metrics logged:
- `train/loss` — Huber (smooth L1) loss per update
- `train/n_updates` — total gradient steps
- `rollout/ep_rew_mean` — mean episodic reward (training, clipped)
- `eval/mean_reward` — mean episodic reward (evaluation, raw scores, every 50k steps)
- `rollout/exploration_rate` — current ε value

---

## Evaluation

### Print stats for 5 episodes

```bash
python evaluate.py \
    --model_path checkpoints/ALE-Breakout-v5/DQN/DQN_final.zip \
    --algorithm DQN \
    --env_id ALE/Breakout-v5 \
    --n_episodes 5
```

Sample output:
```
============================================================
  Evaluation Results
============================================================
  Algorithm      : DQN
  Environment    : ALE/Breakout-v5
  Model          : checkpoints/.../DQN_final.zip
  Episodes run   : 5
============================================================
  Mean Score     :     412.40
  Std  Score     :      87.32
  Min  Score     :     310.00
  Max  Score     :     534.00
  Mean Length    :       3218 steps
============================================================
```

### Record a gameplay video

```bash
python evaluate.py \
    --model_path checkpoints/ALE-Breakout-v5/DoubleDQN/DoubleDQN_final.zip \
    --algorithm DoubleDQN \
    --env_id ALE/Breakout-v5 \
    --record_video \
    --video_dir videos/DoubleDQN
```
```bash

python evaluate.py --model_path checkpoints\ALE-Pong-v5\DQN\DQN_best\best_model.zip --algorithm DQN --env_id ALE/Pong-v5 --record_video --video_dir videos/DQN --n_episodes 1 

# DoubleDQN
python evaluate.py --model_path checkpoints/ALE-Breakout-v5/DoubleDQN/001/DoubleDQN_final.zip --algorithm DoubleDQN --env_id ALE/DonkeyKong-v5 --record_video --video_dir videos/DoubleDQN --n_episodes 6

# DuelingDQN
python evaluate.py --model_path checkpoints\ALE-DonkeyKong-v5\DuelingDQN\001\DuelingDQN_step_3200000_steps.zip --algorithm DuelingDQN --env_id ALE/DonkeyKong-v5 --record_video --video_dir videos/DuelingDQN --n_episodes 5

```

The MP4 is saved to `videos/DoubleDQN/DoubleDQN_ALE-Breakout-v5-*.mp4`.

---

## Reproducing the Benchmark (all three algorithms)

Run the three training jobs sequentially (or on different machines in parallel):

```bash
for algo in DQN DoubleDQN DuelingDQN; do
    python train.py --algorithm $algo
done
```

Then evaluate and collect results:

```bash
for algo in DQN DoubleDQN DuelingDQN; do
    python evaluate.py \
        --algorithm $algo \
        --model_path checkpoints/ALE-Breakout-v5/$algo/${algo}_final.zip \
        --env_id ALE/Breakout-v5 \
        --n_episodes 30 >> results_breakout.txt
done
```

---

## Algorithm Implementation Notes

### Standard DQN (`StandardDQN`)
`StandardDQN` is a thin subclass of SB3's `DQN`.  SB3's default `train()` method computes the Bellman target using the **target network only**:

```
y = r + γ · max_{a'} Q_target(s', a')
```

### Double DQN (`DoubleDQN`)
`DoubleDQN.train()` overrides only the target-computation block.  All other mechanics (replay buffer, ε-schedule, Polyak target updates) are inherited:

```
a* = argmax_{a'} Q_online(s', a')    ← online network selects
y  = r + γ · Q_target(s', a*)        ← target network evaluates
```

### Dueling DQN (`DuelingDQN`)
`DuelingDQN` inherits `DoubleDQN`'s update rule and uses `DuelingCnnPolicy` which swaps SB3's flat Q-head (`QNetwork`) for `DuelingQNetwork`:

```
                  ┌── Value stream ──► V(s)     [512 → 1]
NatureCNN (512) ──┤
                  └── Advantage stream ► A(s,a) [512 → n_actions]

Q(s,a) = V(s) + A(s,a) - mean_{a'}[A(s,a')]
```

---

## Hyperparameter Defaults (DeepMind Nature DQN baseline)

| Parameter | Default | Notes |
|---|---|---|
| `learning_rate` | 2.5e-4 | Adam optimiser |
| `batch_size` | 32 | Mini-batch from replay buffer |
| `buffer_size` | 100,000 | Reduce if OOM; Nature paper used 1M |
| `gamma` | 0.99 | Discount factor |
| `target_update_interval` | 1,000 | Steps between target-net syncs |
| `learning_starts` | 50,000 | Random warm-up before first update |
| `exploration_fraction` | 0.10 | Fraction of training over which ε decays |
| `exploration_final_eps` | 0.01 | Final ε for near-greedy behaviour |
| `train_freq` | 4 | Steps between gradient updates |
| `max_grad_norm` | 10.0 | Gradient clipping |

---

## Troubleshooting

**`RuntimeError: CUDA is not available`**
: Re-install PyTorch with the correct CUDA index URL. Run `nvidia-smi` to verify the driver.

**`ModuleNotFoundError: No module named 'ale_py'`**
: Run `pip install ale-py gymnasium[atari] gymnasium[accept-rom-license]`.

**Out of GPU memory**
: Lower `buffer_size` (replay buffer lives on CPU RAM, not GPU) or `batch_size`. The CNN and experience are typically modest (<2 GB VRAM for a single env).

**Training is slow / CPU-bound**
: Atari throughput is often limited by environment stepping, not the GPU. Use `n_envs: 4` in config.yaml for a 3–4× speedup if your RAM allows.

**TensorBoard shows no curves**
: Ensure you start TensorBoard *from* the `atari_benchmark/` directory, or use `--logdir` with the absolute path.

---

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with Double Q-learning.* AAAI, 30(1).
3. Wang, Z., et al. (2016). *Dueling network architectures for deep reinforcement learning.* ICML, 1995–2003.
