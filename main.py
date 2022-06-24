import gym
from agent import Agent
import torch
from torch import nn

#Example environment

'''
env = gym.make("CartPole-v1")
observation = env.reset()
num_actions = env.action_space.n
'''
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


#Training and Testing
    
'''
#Example model
model = LayeredModel(in_var = env.observation_space.shape[0], final_var = env.action_space.n, layer_count = 2)
#Agent and environment binding
agent = Agent(model = model, gamma = 0.99, lr = 3e-4, train = True)
agent.register_env(env)

#Do training
agent.brain.log_variance = True
agent.train_for(480, respect_cycle = True) # train for 120 seconds
#Test
agent.test()
'''
