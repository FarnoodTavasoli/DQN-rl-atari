for duelingDQN had to increase the exploration rate compared to DQN and DDQN, why ? the first random initial state sometimes happens to be a NOOP (seed rand 42 in pong) this results in the before training data to be heavily scewed towards NOOP actions,  

3.  Buffer (25k slots/env × 8 envs = 200k total) turns over completely by step 200k.
  From that point every sample is (s, NOOP, -1, s').

4. The advantage gradient from gather(Q, action) has this asymmetry:
    chosen action (NOOP): grad = +(n−1)/n = +5/6  → reinforces NOOP
    every other action:   grad =    −1/n  = −1/6  → suppresses others

5. Value stream correctly learns V(s) ≈ −9.43 (the expected return under NOOP).
  Advantage stream permanently converges to: A(s, NOOP) >> A(s, all others).
  Greedy policy = NOOP forever. Evaluation = deterministic NOOP = −21 ± 0.
