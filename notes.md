for duelingDQN had to increase the exploration rate compared to DQN and DDQN, why ? the first random initial state sometimes happens to be a NOOP (seed rand 42 in pong) this results in the before training data to be heavily scewed towards NOOP actions,  

3.  Buffer (25k slots/env × 8 envs = 200k total) turns over completely by step 200k.
  From that point every sample is (s, NOOP, -1, s').

4. The advantage gradient from gather(Q, action) has this asymmetry:
    chosen action (NOOP): grad = +(n−1)/n = +5/6  → reinforces NOOP
    every other action:   grad =    −1/n  = −1/6  → suppresses others

5. Value stream correctly learns V(s) ≈ −9.43 (the expected return under NOOP).
  Advantage stream permanently converges to: A(s, NOOP) >> A(s, all others).
  Greedy policy = NOOP forever. Evaluation = deterministic NOOP = −21 ± 0.

------

This combination (Q-values that never stabilize + a policy that alternates between "dies fast" and "stalls forever" + late-training regression) is the textbook symptom of overestimation-driven oscillation in vanilla DQN, and it's being made worse by two hyperparameters in your config:

target_update_interval: 1000 — SB3/DeepMind's standard is 10,000. Copying the target network 10x more often than usual removes most of the stabilizing lag that keeps bootstrapped targets from chasing themselves.
learning_rate is held flat at 2.5e-4 for the entire 20M-step run (confirmed by the flat learning-rate panel) — nothing damps the step size down once the policy is already decent, so a bad update late in training can kick a good policy into a worse basin with no way to settle back down.
