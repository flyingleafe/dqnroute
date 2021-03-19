import random
import numpy as np
from collections import defaultdict


class AbstractMemory:
    def add(self, sample):
        pass

    def sample(self, n):
        pass


class Memory(AbstractMemory):
    samples = []
    sidx = 0

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append((self.sidx, sample))
        self.sidx += 1

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedMemory(AbstractMemory):
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class PPOMemory(AbstractMemory):
    def __init__(self, batch_size):
        self.addr_idxs = []
        self.dst_idxs = []
        self.action_idxs = []
        self.probs = []
        self.q_estimates = []
        self.neighbours = []
        self.rewards = []
        self.values = []

        self.batch_size = batch_size

    def sample(self):
        batches_starts = np.arange(0, len(self.rewards), self.batch_size)

        idxs = np.arange(0, len(self.rewards))
        np.random.shuffle(idxs)

        batches = np.array([idxs[i:i + self.batch_size] for i in batches_starts])

        return \
            np.array(self.addr_idxs),\
            np.array(self.dst_idxs),\
            np.array(self.action_idxs),\
            np.array(self.probs),\
            np.array(self.q_estimates),\
            self.neighbours,\
            np.array(self.rewards),\
            np.array(self.values), \
            batches

    # def add(self, addr_idx, dst_idx, action_idx, prob, q_estimate, neighbours, reward, v_func):
    def add(self, sample):
        addr_idx, dst_idx, action_idx, prob, q_estimate, neighbours, reward, v_func = sample
        self.addr_idxs.append(addr_idx)
        self.dst_idxs.append(dst_idx)
        self.action_idxs.append(action_idx)
        self.probs.append(prob)
        self.q_estimates.append(q_estimate)
        self.neighbours.append(neighbours)
        self.rewards.append(reward)
        self.values.append(v_func)

    def clear_memory(self):
        self.addr_idxs = []
        self.dst_idxs = []
        self.action_idxs = []
        self.probs = []
        self.q_estimates = []
        self.neighbours = []
        self.rewards = []
        self.values = []

    def __len__(self):
        return len(self.rewards)


class ReinforceMemory(AbstractMemory):
    def __init__(self):
        super(ReinforceMemory, self).__init__()
        # TODO implement
