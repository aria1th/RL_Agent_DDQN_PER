# RL_Agent_DDQN_PER
Gym Agent for Reinforcement Learning, with DDQN + PER implementation with pytorch.

For pure DDQN / PER code, see https://github.com/aria1th/DDQN_with_PER

Pytorch CUDA - CPU context changing is somewhat dirty, also with batches and squeezing, unsqueezing.... deal with it


# How to use

For example code, it chooses Cartpole lab (gym v1) and trains from scratch for 480 seconds(well, actually somewhat more)

Following code constructs gym environment, and initializes Agent that trains / decides / etc for everything else, and binds environment.

```
env = gym.make("CartPole-v1")
agent = Agent(model = model, gamma = 0.99, lr = 3e-4, train = True)
agent.register_env(env)
```

for training your model, just simply:
`agent.train_for(480, respect_cycle = True) # train for 120 seconds`

Agent can stop training with time_limit (seconds), epoch_limit (epochs), episode_limit(game episodes).

for testing your model, just simply:
`agent.test()`
will test your model.
