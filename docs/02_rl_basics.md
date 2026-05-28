# RL Basics (Beginner)

## Core concepts
- State: the observation the agent sees each step.
- Action: what the agent can do (buy, sell, hold).
- Reward: numeric feedback for behavior quality.
- Policy: the mapping from state to actions.
- Episodes: sequences of steps until termination.
- Exploration vs exploitation: trying new actions vs using known good ones.
- PPO intuition: learns a stable policy by limiting large policy updates.

## Why PPO first
PPO is robust, well-tested, and relatively easy to tune. It is a good baseline for learning and benchmarking.

## Common beginner mistakes
- Wrong reward shaping that incentivizes risk
- Data leakage between train and test
- Training for too long without evaluation
