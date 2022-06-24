import gym
from agent import Agent
import torch
from torch import nn



class LayeredModel(nn.Module):
    def __init__(self, in_var = 1, final_var = 4, features = 40, layer_count = 1, p = 0.35, act = nn.ReLU):
        super().__init__()
        self.p = p
        self.act = act
        layers = [nn.Linear(in_var, features),]
        for i in range(layer_count):
            layers.extend(self._layer(features))
        self.layers = nn.Sequential(*layers)
        self.layers.append(nn.Linear(features, final_var))
        
    def forward(self, x):
        return self.layers(x)

    def _layer(self, features):
        return [nn.Linear(features, features), self.act(), nn.Dropout(self.p)]

env = gym.make("CartPole-v1")
observation = env.reset()
num_actions = env.action_space.n

#Example model
model = LayeredModel(in_var = env.observation_space.shape[0], final_var = env.action_space.n, layer_count = 2)

#Example Agent
agent = Agent(model = model, gamma = 0.99, lr = 3e-4, train = True)

#Bind Environment
agent.register_env(env)

#Logging
agent.brain.log_variance = True

#Training
agent.train_for(480, respect_cycle = True) # train for 480 seconds

#Testing
agent.test()

