import numpy as np

import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import namedtuple
import random
from . import per

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'), defaults = (None,) * 5)
import random

class Brain:
    '''This is cleaned-up code of original DDQN & PER implementation. '''
    def __init__(self, num_actions, batch_size, gamma, replay_memory, model,tau : float = 0.0, optimizer = None, update_freq : int = 2000,
                 observe_scenarios : int = 0, scheduler = None, log_variance = True, ddqn = True, device = None, use_per = True,
                 eps_annealing = 0):
        '''
        Brain(Actions, Batch size, Gamma, Replay memory object, Model,tau(update ratio, float), Optimizer(optional), Update frequency(default = 2000),
            Scenarios to observe(default = 0), Scheduler(Optional), Logging Toggle(default = True), DDQN Boolean(default = True),
            device (default = Auto cuda / cpu), use_per(default = True, PER memory object), eps_annealing(default = None) )
        Replay memory should implement memory.clear, memory.__len__, memory.sample(batch size), memory.update(optional for ddqn)
        Brain.model is 'exposed' model, for use in external or internal.
        Brain.target_model is not exposed model, it should not be accessed externally.
        PER is on at default, it uses memory.update(idx, error) method.

        eps_annealing : Object that has get(episode : int or None) -> float method. If its None, default epsilon-greedy search will be used.
        If its int 0: it will use default implemented annealing.

        '''
        self.num_actions = num_actions #input something -> final num_actions
        self.logger_copy = [] #log variance
        self.batch_size = batch_size
        self.gamma = gamma #current reward + future 1 step forward reward * gamma
        self.memory = replay_memory #Replay memory object
        if type(self.memory) is int or type(self.memory) is float:
            self.memory = per.PrioritizedMemory(int(self.memory))
        self.tau = tau
        self.ddqn = ddqn
        self.optimizer = optimizer
        self.log_variance = log_variance
        self.observes = observe_scenarios
        self.counter = 0
        self.update_freq = update_freq #will update target model
        self.prior = use_per
        assert self.memory.clear is not None
        assert len(self.memory) == 0
        assert self.memory.sample is not None
        if ddqn:
            assert self.memory.update is not None
        self.target_model = model #assign reference, we will use model directly
        if device is None: #auto
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = deepcopy(model) #learn here and copy to model, for safety create new model.
        self.model.to(device)
        if not self.ddqn:
            self.target_model = self.model #by reference, normal dqn uses same object
        self.target_model.to(device)
        self.soft_update(tau = 1.0)
        if self.ddqn:
            for params in self.model.parameters():
                params.require_grad = False #detach grad if its ddqn

        self.target_optimizer = None
        self.scheduler = scheduler
        
        self.debug = False

        self.eps_scheduler = eps_annealing
        if eps_annealing == 0:
            self.eps_scheduler = EpsilonAnnealing(observe = self.observes)

    def hard_update(self, tau = 0):
        '''hard update : loads from state dict, No interpolation'''
        if self.log_variance:
            print('Copying values to model(Hard update)...')
            variance = sum((x - y).abs().sum() for x, y in zip(self.target_model.state_dict().values(), self.model.state_dict().values()))
            self.logger_copy.append(variance.item())
            print(variance.item())
        self.model.load_state_dict(self.target_model.state_dict())
    def soft_update(self, tau = 0.95):
        '''soft update : interpolates values w.r.t. tau, if tau == 0 then direct copy'''
        if self.log_variance:
            print('Copying values to model softly...')
            # copy, target -> model data
            variance = sum((x - y).abs().sum() for x, y in zip(self.target_model.state_dict().values(), self.model.state_dict().values()))
            self.logger_copy.append(variance.item())    
            print(variance.item())
        for model_param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            model_param.data.copy_(tau * model_param.data + (1.0 - tau) * target_param.data)
            
    def replay(self):
        '''learns from memory'''
        if len(self.memory) < self.batch_size:
            return
        self.counter += 1
        if self.counter % self.update_freq == 0:
            if self.tau == 0: #hard update
                self.hard_update(tau = 0)
            else:
                self.soft_update(tau = tau)
        self.replay_new()
        
    def replay_new(self):
        '''Actual training code'''
        #First, extract from memory.
        if self.prior:
            data, idxs, b =  self.memory.sample(self.batch_size)
        else:
            data = self.memory.sample(self.batch_size)
            idxs,b = None, None
        transition = Transition(*zip(*data)) #actually its better to have direct batches from memory.
        device = self.device
        state_batch = torch.cat(transition.state).to(device)
        action_batch = torch.cat(transition.action).to(device).unsqueeze(1) #we have simple flat list of actions -> convert it to [[],[],....]
        next_state_batch = torch.cat(transition.next_state).to(device)
        reward_batch = torch.cat(transition.reward).to(device)
        done_mask = torch.ByteTensor(transition.done).to(device) #tensor([a,b,c,d])
        #Evaluation
        self.target_model.eval()
        
        if self.debug: #debug code, print objects
            print(state_batch)
            print(action_batch)
            print(next_state_batch)
            print(reward_batch)
            print(done_mask)
        q_values = self.target_model(state_batch)
        q_state_action = q_values.gather(1, action_batch) #q values for current state
        q_state_action = q_state_action.squeeze() # tensor([a,b,c,d]) in device
        if self.debug:
            print(q_state_action)
        
        if self.ddqn:
            #DDQN : use model / target model separately
            self.model.eval()
            q_next_state_action = self.model(next_state_batch).detach() #q values for next state but fixed model finds it
            _, a_prime = q_next_state_action.max(1)
            q_target_action_value = q_next_state_action.gather(1, a_prime.unsqueeze(1)).squeeze() #q values for selected(wanted) actions from fixed model
            q_target_action_value = (1 - done_mask) * q_target_action_value #but done mask nullifies it
            q_gamma_regard = reward_batch + self.gamma * q_target_action_value # reward is now adjusted w.r.t gamma
            #or loss(q_state_action, reward_batch + self.gamma * q_target_action_value)
            if self.debug:
                print(q_next_state_action)
                print(q_target_action_value)
                print(self.gamma)
                print(q_gamma_regard)
                print(reward_batch)
                print(q_state_action) #we want q state action(from target model) to be close to gamma_regard, 
        else:
            #Vanilla DQN : use target model in everything
            q_next_state_action = self.target_model(next_state_batch).detach()
            q_target_action_value, a_prime = q_next_state_action.max(1)
            q_target_action_value = (1 - done_mask) * q_target_action_value
            q_gamma_regard = reward_batch + self.gamma * q_target_action_value
        #Extracted new Q Values to train. I.E. If action leaded to death, reward should be lower than previous.
            
        if self.prior:
            #PER then update errors, if error is big, value should be corrected quickly.
            q_state_action_pure = q_state_action.detach()
            q_error = (1 - done_mask) * torch.abs(q_target_action_value - q_state_action_pure)
            for i in range(self.batch_size):
                if done_mask[i]: #Nothing to update if it was final action
                    continue
                idx = idxs[i]
                self.memory.update(idx, q_error[i].item(), update_gamma = 0) #update_gamma is Interpolation Value. if you think more intense learning is required, use value from
                #0 to 1

        #Training model
        self.target_model.train()
        self.optimizer.zero_grad()
        #Choose loss type by yourself.
        loss = F.smooth_l1_loss(q_state_action, q_gamma_regard)
        #or (q_gamma_regard - q_state_action).backward()
        if self.debug:
            print(loss)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def biasedRandom(self, reverse = -1):
        bias = 4
        if random.random() < 1 / bias:
            return int(0.5 - 0.5 * reverse)
        else:
            return int(0.5 + 0.5 * reverse)
        
    def decide_action(self, state, episode = None, **kwargs):
        #Single state decision. For multiple states, modify this code.
        if episode is None or episode < self.observes:
            #Observe
            epsilon = 0.
        else:
            #Epsilon annealing, optional
            if self.eps_scheduler is not None:
                epsilon = self.eps_scheduler.get(episode)
            else:
                epsilon = 1 / (episode - self.observes + 1)
        if epsilon <= np.random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state.to(self.device)).argmax(1).view(1,1)
        else:
            action = torch.randint(0, self.num_actions, (1,1), device = self.device)
        return action
    def decide_actions(self, states, episodes = None):
        ''' State batch + (Episodes batch -> Epsilon batch) -> Decided actions batch'''
        if self.eps_scheduler is not None:
            eps_batch = self.eps_scheduler.get_batch(episodes)
        elif episodes is not None:
            eps_batch = 1 / (episode - self.observes + 1)
        else:
            eps_batch = torch.zeros((states.shape[0],1))
        if eps_batch == 0: #pick from model
            self.model.eval()
            with torch.no_grad():
                actions = self.model(states.to(self.device)).argmax(1).view(states.shape[0], 1)
            return actions
        actions = torch.where(eps_batch <= random.random(), self.model(states.to(self.device)).argmax(1).view(states.shape[0], 1),
                           torch.randint(0, self.num_actions, size = (states.shape[0],1)).view(states.shape[0],1))
        return actions.to(self.device)
    def memorize(self, state, action, next_state, reward, done):
        device = self.device
        model = self.model
        target_model = self.target_model
        model.eval()
        target = model(state).data
        old_val = target[0][action]
        target_model.eval()
        target_val = target_model(next_state).data
        if done:
            newval= reward
        else:
            newval = reward + self.gamma * torch.max(target_val)
        err = abs(old_val - newval)
        self.memory.push(err, state, action, reward, next_state, done)
            
    
class EpsilonAnnealing:
    def __init__(self, min_val = 1e-7, observe = 0, func = None):
        self.min_val = 1e-7
        self.observe = observe
        self.func = func #function of vectorized episode(int)-> epsilon(float)
    def get(self, episode = None):
        if episode is None:
            return 0
        return max(self.min_val, 1 / (episode - self.observe + 1))
    def get_batch(self, episode = None):
        ''' episodes as Numpy.ndarray or similar forms. nxm shape -> nxm shape epsilons. Episode should be integer.'''
        if episode is None:
            return 0
        if self.func is not None:
            return self.func(episode)
        return torch.where(episode <= self.observe, 0, (1 / (episode - self.observe + 1)).clip(self.min_val))
