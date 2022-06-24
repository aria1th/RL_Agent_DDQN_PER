from ddqn_brain import brain_ddqn
import time
from scheduler import customScheduler
import torch
from torch import optim

class Agent:
    def __init__(self, model = None, num_actions = 2, lr = 5e-3, batch_size = 32, gamma = 0.99, repeat = 32, observe = 300, train = True, device = None, optimizer = optim.Adam):
        self.train = train
        self.learning_rate = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.action = None
        self.repeat = repeat
        self.observe = observe
        if device is None: #auto
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if model is None:
            try:
                self.model = model_structure.reload().to(self.device)
            except:
                raise Exception( 'Model structure is not defined and No models present!')
        else:
            self.model = model
        self.optimizer = optimizer( self.model.parameters(), lr=self.learning_rate )
        
        self.scheduler = customScheduler.CosineAnnealingWarmupRestarts(self.optimizer, 200, 1.2, max_lr = self.learning_rate, min_lr = 1e-7, gamma = self.gamma)
        
        self.env = None


        self.brain = brain_ddqn.Brain(num_actions, self.batch_size, self.gamma, model = self.model,replay_memory = 3e6, observe_scenarios = self.observe, optimizer = self.optimizer,scheduler = self.scheduler,
                                      log_variance = False, device = self.device)
        
        self.state = None
        self.scenario_num = 0

        self.flag = False # for cosine annealing stop signal
        self.may_stop = False
    def learn(self):
        if self.may_stop:
            return None #do not learn anytihng when it should stop.
        for _ in range(self.repeat):
            self.brain.replay()
            if self.flag and self.scheduler.is_EOC():
                self.may_stop = True
                break
            
    def memorize(self, state, action, next_state, reward, done):
        self.brain.memorize(state, action, next_state, reward, done)
        
    def get_action(self, state, episode, asItem = False):
        obj =  self.brain.decide_action(state, episode)
        if asItem:
            return obj.item()
        return obj
    
    def sequence(self):
        if self.scenario_num >= self.observe and self.train:
            self.learn()
        if self.env is not None:
            if self.action is None:
                self.action = self.env.action_space.sample()
            next_observe, reward, done, info = self.env.step(self.action)
            if not done:
                next_state = self.preprocess(next_observe)
            else:
                next_state = self.state #reuse state
                
            reward = torch.tensor([reward], device = self.device, dtype = torch.float32)
            if self.state is not None and self.action is not None:
                self.memorize(self.state, torch.tensor([self.action,], device = self.device), next_state, reward, done)
            self.state = next_state
            self.action = self.get_action(self.state, self.scenario_num).item()
            if done:
                self.flip_episode()
            
        return None
    
    def train_for(self, time_limit = None, epoch_limit = None, episode_limit = None, respect_cycle = False):
        '''train for given time or epochs
            Cycle respect  : cosineAnnealingWarmupRestarts converges at small local minima and then repeats, so it respects and delays stop'''
        st = time.time()
        epochs = 0
        while True:
            epochs += 1
            self.sequence()
            if time_limit is not None and time.time() - st > time_limit:
                if respect_cycle:
                    self.flag = True
                    if self.may_stop:
                        break
                else:
                    break
            if epoch_limit is not None and epochs > epoch_limit:
                if respect_cycle:
                    self.flag = True
                    if self.may_stop:
                        break
                else:
                    break
            if episode_limit is not None and self.scenario_num >= episode_limit:
                if respect_cycle:
                    self.flag = True
                    if self.may_stop:
                        break
                else:
                    break
        self.may_stop = False
        self.flag = False
        return epochs
    
    def test(self, render_skip : float = 1, max_time = 1e4):
        next_observe = self.env.reset()
        #cartpole, does not have info
        next_observe = self.preprocess(next_observe)
        render_skip = int(max(1, render_skip))
        action = self.get_action(next_observe, None).item()
        done = False
        counts = 0
        while not done:
            next_observe, reward, done, info = self.env.step(action)
            if counts % render_skip == 0:
                self.env.render()
            #preprocess
            next_observe = torch.tensor(next_observe, device = self.device, dtype = torch.float32).unsqueeze(0)
            action = self.get_action(next_observe, None).item()
            counts += 1
            if counts > max_time:
                break
        return counts
    
    def preprocess(self, obj, dtype = torch.float32):
        return torch.tensor(obj, device = self.device, dtype = dtype).unsqueeze(0)
    
    def register_env(self, environment):
        self.env = environment
        
    def flip_episode(self):
        '''flips episode and set state to initial observation'''
        self.state = self.preprocess(self.env.reset())
        self.scenario_num += 1
        self.action = None
