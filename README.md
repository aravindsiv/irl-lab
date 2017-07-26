# irl-lab
A WIP implementation of popular Inverse Reinforcement Learning algorithms for various tasks. Suggestions are welcome for more complex environments and algorithms, as well as possible bugs in the current implementation.

The code in this repository is heavily inspired by [this](https://github.com/MatthewJA/Inverse-Reinforcement-Learning) work by Matthew Alger, while the idea itself (not to mention, the name) is inspired by OpenAI and UC Berkeley's [rllab](https://github.com/openai/rllab).

## Environments
The following environments have been implemented:
* Gridworld

## Algorithms
The following algorithms have been implemented:
* Policy Iteration
* [Relative Entropy Inverse Reinforcement Learning](http://proceedings.mlr.press/v15/boularias11a/boularias11a.pdf)

## Sample Usage
```python
from env.GridWorld import GridWorld
from algo.PolicyIteration import PolicyIteration
from algo.RelEntIRL import RelEntIRL

gw = GridWorld(10)
print gw.rewards
# Obtain the optimal policy for the environment to generate expert demonstrations
pi = PolicyIteration(gw)
optimal_policy = pi.policy_iteration(100)

expert_demos = gw.generate_trajectory(optimal_policy)
nonoptimal_demos = gw.generate_trajectory()
relent = RelEntIRL(expert_demos, nonoptimal_demos)
# Train the model with default hyperparameters. 
relent.train()
print relent.weights.reshape(10,10)
```
