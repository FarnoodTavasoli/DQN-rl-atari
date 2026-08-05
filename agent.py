"""
agent.py — DQN Variant Factory
================================
Implements three Q-learning algorithms as Stable-Baselines3-compatible classes:

  ┌─────────────┬────────────────────────────────────────────────────────────┐
  │ Class        │ Description                                               │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ StandardDQN  │ Mnih et al. (2015). Target network selects AND evaluates  │
  │              │ the greedy action.  SB3's DQN default.                    │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ DoubleDQN    │ van Hasselt et al. (2016). Online network selects the      │
  │              │ greedy action; target network evaluates it.  Reduces       │
  │              │ overestimation bias.                                       │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ DuelingDQN   │ Wang et al. (2016) + Double Q-learning.  Separate value   │
  │              │ V(s) and advantage A(s,a) streams computed by             │
  │              │ DuelingQNetwork, aggregated as:                            │
  │              │   Q(s,a) = V(s) + A(s,a) - mean_{a'}[A(s,a')]            │
  └─────────────┴────────────────────────────────────────────────────────────┘

Public entry-point
------------------
  create_agent(config, env) → one of the three agent types, fully initialised
  and ready for .learn().

Architecture notes
------------------
* DoubleDQN overrides only the Bellman target computation inside train().
  All other SB3 mechanics (replay buffer, exploration schedule, target-network
  Polyak updates via _on_step(), TensorBoard logging) are inherited unchanged.

* DuelingQNetwork replaces QNetwork's flat MLP head with two parallel streams.
  The parent's placeholder `q_net` attribute is replaced with nn.Identity()
  so that SB3's internal attribute checks are satisfied while keeping the
  parameter registry clean for the optimiser.

* DuelingDQN inherits DoubleDQN so it automatically uses the decoupled
  action-selection / action-evaluation target.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.policies import CnnPolicy, QNetwork
from torch import nn

# =============================================================================
# Dueling Network Components
# =============================================================================


class DuelingQNetwork(QNetwork):
    """
    Q-Network with a Dueling architecture (Wang et al., 2016).

    After the shared convolutional feature extractor (NatureCNN), the
    representation is split into two independent streams:

      Value stream      V(s)     → scalar  — "how good is this state?"
      Advantage stream  A(s, a)  → vector  — "how much better is action a?"

    They are recombined using the mean-subtraction aggregation from the paper:

        Q(s, a) = V(s) + [ A(s, a) − (1/|A|) Σ_{a'} A(s, a') ]

    The mean subtraction forces the value stream to learn a unique V(s) and
    prevents the advantage stream from being off by an arbitrary constant.

    Parameters
    ----------
    observation_space : spaces.Space
        Environment observation space (after preprocessing).
    action_space : spaces.Discrete
        Discrete action space.
    features_extractor : nn.Module
        Pre-built convolutional feature extractor (NatureCNN).
    features_dim : int
        Output dimensionality of the feature extractor (default 512 for NatureCNN).
    net_arch : list[int] | None
        Not used by the dueling head; kept for API compatibility.
    activation_fn : type[nn.Module]
        Not used directly; kept for API compatibility.
    normalize_images : bool
        Whether to normalise pixel observations to [0, 1].
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        action_dim = int(self.action_space.n)

        # Discard the flat MLP head created by the parent class and replace
        # it with an nn.Identity placeholder so that:
        #   (a) SB3 attribute checks on `.q_net` do not raise AttributeError.
        #   (b) The unused MLP parameters are removed from the registry,
        #       keeping the optimiser and parameter count clean.
        self.q_net = nn.Identity()

        # ---- Value stream  V(s) : features_dim → 512 → 1 ------------------
        self.value_stream = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # ---- Advantage stream  A(s,a) : features_dim → 512 → n_actions ----
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Compute Q-values for all actions from a batch of observations.

        Parameters
        ----------
        obs : th.Tensor
            Batch of preprocessed observations, shape ``(B, C, H, W)``.

        Returns
        -------
        th.Tensor
            Q-value tensor of shape ``(B, n_actions)``.
        """
        # Shared convolutional features — shape (B, features_dim)
        features = self.extract_features(obs, self.features_extractor)

        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, n_actions)

        # Dueling aggregation (Wang et al., 2016, Eq. 9)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values  # (B, n_actions)


class DuelingCnnPolicy(CnnPolicy):
    """
    CnnPolicy that substitutes the standard QNetwork head with DuelingQNetwork.

    Passed as the `policy` argument when constructing DuelingDQN.  Overrides
    only `make_q_net()` so that both the online and target networks are built
    as DuelingQNetwork instances; all other policy logic is inherited.
    """

    def make_q_net(self) -> DuelingQNetwork:
        """
        Instantiate a DuelingQNetwork.

        Called twice by DQNPolicy._build(): once for `q_net` (online) and
        once for `q_net_target`.  Each call creates a fresh features extractor
        so that the two networks have independent weights.

        Returns
        -------
        DuelingQNetwork
            Fully initialised dueling Q-network moved to the correct device.
        """
        # _update_features_extractor clones the features extractor kwargs so
        # that online and target networks own independent CNN weights.
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return DuelingQNetwork(**net_args).to(self.device)


# =============================================================================
# Algorithm Wrappers
# =============================================================================


class StandardDQN(DQN):
    """
    Standard DQN — Mnih et al. (2015).

    The Bellman target is computed entirely by the *target* network:

        y = r + γ · max_{a'} Q_target(s', a')

    Both action selection (argmax) and evaluation (Q-value lookup) use the
    same frozen target network.  This override of train() mirrors SB3's
    default logic exactly while adding extra diagnostic metrics to TensorBoard.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: List[float] = []
        max_q_values: List[float] = []
        td_errors: List[float] = []
        grad_norms: List[float] = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # Vanilla DQN: target network both selects and evaluates
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.max(dim=1, keepdim=True).values
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # All Q-values for the current obs (needed for max-Q statistic)
            all_current_q = self.q_net(replay_data.observations)
            max_q_values.append(all_current_q.max(dim=1).values.mean().item())
            current_q_values = th.gather(
                all_current_q, dim=1, index=replay_data.actions.long()
            )  # (B, 1)

            assert current_q_values.shape == target_q_values.shape

            td_errors.append(
                (current_q_values - target_q_values).abs().mean().item()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.max_grad_norm
            ).item()
            grad_norms.append(grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/mean_max_q", np.mean(max_q_values))
        self.logger.record("train/mean_td_error", np.mean(td_errors))
        self.logger.record("train/grad_norm", np.mean(grad_norms))


class DoubleDQN(DQN):
    """
    Double DQN — van Hasselt, Guez & Silver (2016).

    Decouples action *selection* from action *evaluation* to address the
    overestimation bias inherent in standard DQN:

        a*  = argmax_{a'} Q_online(s', a')        ← online network selects
        y   = r + γ · Q_target(s', a*)            ← target network evaluates

    All other mechanics (replay buffer, exploration, Polyak target updates
    triggered by _on_step(), TensorBoard logging) are inherited from DQN.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Execute ``gradient_steps`` gradient updates using the Double DQN target.

        Parameters
        ----------
        gradient_steps : int
            Number of gradient updates to perform in this call.
        batch_size : int
            Number of transitions sampled from the replay buffer per update.
        """
        # Activate batch-norm / dropout in training mode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: List[float] = []
        max_q_values: List[float] = []
        td_errors: List[float] = []
        grad_norms: List[float] = []

        for _ in range(gradient_steps):
            # ---- Sample replay buffer ---------------------------------------
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # ---- DOUBLE DQN TARGET COMPUTATION -------------------------
                # Step 1 — Online network selects the greedy action for s'
                #           (decouples selection from evaluation)
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B, 1)

                # Step 2 — Target network evaluates the selected action
                #           (provides a stable, less biased Q-value estimate)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)  # (B, 1)

                # 1-step TD target  y = r + γ(1 − done) · Q_target(s', a*)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # ---- Current Q-values for the actions taken --------------------
            # Keep all_current_q before gather to compute the max-Q statistic
            all_current_q = self.q_net(replay_data.observations)
            max_q_values.append(all_current_q.max(dim=1).values.mean().item())
            current_q_values = th.gather(
                all_current_q, dim=1, index=replay_data.actions.long()
            )  # (B, 1)

            assert current_q_values.shape == target_q_values.shape, (
                f"Shape mismatch: current={current_q_values.shape}, "
                f"target={target_q_values.shape}"
            )

            # ---- Huber loss (robust to outlier Q-value errors) -------------
            td_errors.append(
                (current_q_values - target_q_values).abs().mean().item()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # ---- Gradient update -------------------------------------------
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip only the online Q-network's gradients; the target network
            # has zero gradients (computed under th.no_grad()) and should not
            # be included to avoid scanning its parameters unnecessarily.
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.max_grad_norm
            ).item()
            grad_norms.append(grad_norm)
            self.policy.optimizer.step()

        # ---- Bookkeeping ---------------------------------------------------
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/mean_max_q", np.mean(max_q_values))
        self.logger.record("train/mean_td_error", np.mean(td_errors))
        self.logger.record("train/grad_norm", np.mean(grad_norms))


class DuelingDQN(DoubleDQN):
    """
    Dueling Double DQN — Wang et al. (2016) + van Hasselt et al. (2016).

    Combines the Dueling Q-Network architecture with Double Q-learning updates.
    The network splits into separate value V(s) and advantage A(s,a) streams
    after the shared convolutional encoder, then recombines them:

        Q(s, a) = V(s) + A(s, a) − mean_{a'} A(s, a')

    The Double Q-learning update rule is inherited from DoubleDQN.train().
    DuelingCnnPolicy is used as the default policy so that both the online
    and target networks are DuelingQNetwork instances.
    """

    def __init__(
        self,
        policy: Union[str, Type[DuelingCnnPolicy]] = DuelingCnnPolicy,
        env: Optional[GymEnv] = None,
        **kwargs: Any,
    ) -> None:
        # If a user accidentally passes the string "CnnPolicy", redirect to
        # the dueling variant so the architecture is never silently wrong.
        if isinstance(policy, str) and policy in ("CnnPolicy", "MlpPolicy"):
            policy = DuelingCnnPolicy
        super().__init__(policy=policy, env=env, **kwargs)


# =============================================================================
# Factory Function
# =============================================================================

_ALGORITHM_REGISTRY: Dict[str, Type[DQN]] = {
    "DQN": StandardDQN,
    "DoubleDQN": DoubleDQN,
    "DuelingDQN": DuelingDQN,
}


def create_agent(config: Dict[str, Any], env: GymEnv) -> DQN:
    """
    Instantiate the agent specified by ``config["algorithm"]``.

    Parameters
    ----------
    config : dict
        Loaded hyperparameter dictionary (from config.yaml).  Must contain
        the key ``"algorithm"`` and all standard DQN hyperparameter keys.
    env : GymEnv
        The (vectorised) training environment returned by make_atari_env().

    Returns
    -------
    DQN
        Fully initialised agent (StandardDQN | DoubleDQN | DuelingDQN).

    Raises
    ------
    ValueError
        If ``config["algorithm"]`` is not one of the registered names.
    """
    algorithm_name: str = config["algorithm"]

    if algorithm_name not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Choose from: {list(_ALGORITHM_REGISTRY.keys())}"
        )

    AlgorithmClass = _ALGORITHM_REGISTRY[algorithm_name]

    # --- Optimizer selection ------------------------------------------------
    # Maps the string name from config["optimizer"] to a (class, kwargs) pair.
    # optimizer_kwargs are merged with any extra keys from config["optimizer_kwargs"].
    _OPTIMIZER_REGISTRY: Dict[str, tuple] = {
        "adam":    (th.optim.Adam,    dict(eps=1e-5)),
        "adamw":   (th.optim.AdamW,   dict(eps=1e-5, weight_decay=1e-4)),
        "rmsprop": (th.optim.RMSprop, dict(eps=1e-5, alpha=0.95)),
        "sgd":     (th.optim.SGD,     dict(momentum=0.9)),
    }

    optimizer_name: str = config.get("optimizer", "adam").lower()
    if optimizer_name not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Choose from: {list(_OPTIMIZER_REGISTRY.keys())}"
        )

    opt_class, opt_kwargs = _OPTIMIZER_REGISTRY[optimizer_name]
    # Allow config["optimizer_kwargs"] to override or extend the defaults
    opt_kwargs = {**opt_kwargs, **config.get("optimizer_kwargs", {})}
    policy_kwargs = dict(optimizer_class=opt_class, optimizer_kwargs=opt_kwargs)

    # --- TensorBoard log directory ------------------------------------------
    # If train.py has already resolved a numbered run directory it stores the
    # full path in config['_tb_log_dir']; fall back to the legacy layout so
    # the function still works when called in isolation.
    env_tag = config["env_id"].replace("/", "-").replace(":", "_")
    lr = config["learning_rate"]
    bs = config["batch_size"]
    tb_log_dir = config.get("_tb_log_dir") or f"{config['log_dir']}/{env_tag}"

    # --- Policy selection ---------------------------------------------------
    # DuelingDQN manages its own policy default internally.
    # StandardDQN and DoubleDQN use SB3's built-in CnnPolicy.
    policy_arg: Union[str, Type[DuelingCnnPolicy]] = (
        DuelingCnnPolicy if algorithm_name == "DuelingDQN" else "CnnPolicy"
    )

    # --- DuelingDQN exploration overrides -----------------------------------
    # The dueling architecture is vulnerable to an "advantage collapse" at the
    # start of training:
    #
    #   1. At random init, A(s,a) − mean(A) ≈ 0 for all actions, so all
    #      Q-values equal V(s).  PyTorch's argmax breaks ties by returning
    #      index 0 (NOOP in Pong), which doesn't move the paddle.
    #
    #   2. With the default exploration_fraction=0.15, epsilon reaches 0.02
    #      by step 150k.  After that the agent is 98% greedy = 98% NOOP.
    #
    #   3. The buffer (200k slots) turns over completely by step ~200k;
    #      every sampled transition is (s, NOOP, −1, s').
    #
    #   4. The advantage gradient is asymmetric:
    #        chosen action (NOOP): grad = +(n−1)/n = +5/6  → reinforces NOOP
    #        every other action:   grad =    −1/n  = −1/6  → suppresses others
    #
    #   5. The advantage stream permanently converges to NOOP-dominant;
    #      the value stream correctly learns V(s) ≈ −21 * discount factor
    #      but the paddle never moves again.
    #
    # Fix: keep epsilon high long enough that the buffer always contains
    # enough non-NOOP transitions to let the advantage stream differentiate
    # actions, and warm-start the buffer with more random data.
    if algorithm_name == "DuelingDQN":
        effective_exploration_fraction = max(config["exploration_fraction"], 0.30)
        effective_learning_starts = max(config["learning_starts"], 50_000)
    else:
        effective_exploration_fraction = config["exploration_fraction"]
        effective_learning_starts = config["learning_starts"]

    agent = AlgorithmClass(
        policy=policy_arg,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        target_update_interval=config["target_update_interval"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        learning_starts=effective_learning_starts,
        exploration_fraction=effective_exploration_fraction,
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_final_eps=config["exploration_final_eps"],
        max_grad_norm=config["max_grad_norm"],
        # Store next_obs implicitly (as a view into the obs ring buffer)
        # instead of duplicating every frame-stacked observation. This
        # roughly halves replay-buffer RAM usage, which is what lets us
        # run a much larger buffer_size within a 16 GB budget.
        # NOTE: SB3's ReplayBuffer forbids combining optimize_memory_usage
        # with handle_timeout_termination, so the latter must be disabled
        # here. In practice this only affects bootstrapping on the rare
        # timestep where an episode is truncated (not terminated) exactly
        # at the max-episode-steps limit -- a negligible effect for Atari.
        optimize_memory_usage=config.get("optimize_memory_usage", True),
        replay_buffer_kwargs=(
            {"handle_timeout_termination": False}
            if config.get("optimize_memory_usage", True)
            else None
        ),
        tensorboard_log=tb_log_dir,
        device="cuda",
        verbose=1,
        seed=config.get("seed", 42),
    )

    est_buffer_gb = (config["buffer_size"] * (28224 if config.get("optimize_memory_usage", True) else 56448)) / 1024 ** 3

    override_note = (
        "  (* exploration and learning_starts overridden for DuelingDQN)\n"
        if algorithm_name == "DuelingDQN" and (
            effective_exploration_fraction != config["exploration_fraction"]
            or effective_learning_starts != config["learning_starts"]
        )
        else ""
    )
    print(
        f"\n{'=' * 60}\n"
        f"  Algorithm : {algorithm_name}\n"
        f"  Policy    : {agent.policy_class.__name__}\n"
        f"  Optimizer : {optimizer_name}  {opt_kwargs}\n"
        f"  Device    : {agent.device}\n"
        f"  N envs    : {config['n_envs']}\n"
        f"  LR        : {lr}\n"
        f"  Batch     : {bs}\n"
        f"  Buffer    : {config['buffer_size']:,}  (~{est_buffer_gb:.1f} GB RAM)\n"
        f"  Expl frac : {effective_exploration_fraction}\n"
        f"  Learn strt: {effective_learning_starts:,}\n"
        f"  TBlog     : {tb_log_dir}\n"
        f"{override_note}"
        f"{'=' * 60}\n"
    )

    return agent
