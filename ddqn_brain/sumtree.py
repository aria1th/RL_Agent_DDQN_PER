
import numpy as np
try:
    from numba import njit
    @njit
    def index(array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[0]
except ImportError:
    print('numba is not found, using natural indexing...[sumtree.py]')
    def index(array, item):
        idx = np.where(array == item)[0]
        if len(idx) == 0:
            return None
        else:
            return idx[0]
# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.set = set()
        self.n_entries = 0
        self.overwrite = False
        self.force_update_when_add = True # in add() function, try updating existing values if this is true

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        if data in self.set:
            return_flag = True
            if self.force_update_when_add:
                idx = index(self.data, data)
                if idx is None:
                    self.set.add(data)
                    return_flag = False
                else:
                    self.update(idx, p, 0) #force update value itself
                    return_flag = True
            if return_flag:
                return
        else:
            self.set.add(data)
        idx = self.write + self.capacity - 1
        if self.overwrite:
            self.set.remove(self.data[self.write])
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.overwrite = True
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p, update_gamma = 0):
        newval = self.tree[idx] * update_gamma + (1-update_gamma) * p
        change = newval - self.tree[idx]

        self.tree[idx] = newval
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
    def __len__(self):
        return self.n_entries

