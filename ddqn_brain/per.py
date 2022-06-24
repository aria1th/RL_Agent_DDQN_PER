import random
try:
    import sumtree
except:
    from . import sumtree
import numpy as np

try:
    import torch
    use_torch = True
except:
    use_torch = False
class PrioritizedMemory:
    e = 0.11
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    update_gamma = 0.95
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = None # Do not access this directly!
        self.tree = sumtree.SumTree(capacity) #see https://github.com/rlcode/per/blob/master/prioritized_memory.py
        self.prior = True
    def _get_priority(self, error):
        return (abs(error) + self.e) ** self.a
    def push(self, error, state, action, reward, state_next, done): 
        #error is reward change so we need to calculate
        p = self._get_priority(error)
        self.tree.add(p, (state, action, state_next, reward, done))
    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a,b)
            (idx, p, data) = self.tree.get(s)
            while data == 0:
                s = random.uniform(0, a)
                (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_prob = priorities / self.tree.total()
        is_w = np.power(self.tree.n_entries * sampling_prob, -self.beta)
        if is_w.max() != 0: is_w /= is_w.max() #no DivisionByZero
        return batch, idxs, is_w
    def __len__(self):
        return len(self.tree)
    def update(self, idx, error, update_gamma = None):
        p = self._get_priority(error)
        if update_gamma is None:
            update_gamma = self.update_gamma
        self.tree.update(idx, p, update_gamma)
    def clear(self):
        self.capacity = capacity
        self.memory = None # Do not access this directly!
        del self.tree
        self.tree = sumtree.SumTree(capacity) #see https://github.com/rlcode/per/blob/master/prioritized_memory.py
        self.prior = True
        self.beta = PrioritizedMemory.beta
