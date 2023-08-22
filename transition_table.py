import torch
import numpy as np
from random import randrange

class TransitionTable(object):

    def __init__(self, transition_params):

        self.transition_params = transition_params

        self.agent_params = transition_params["agent_params"]
        self.gpu = self.agent_params["dqn_config"].gpu
        self.discount = self.agent_params["discount"]
        self.state_size = self.agent_params["dqn_config"].state_size
        self.n_step_n = self.agent_params["n_step_n"]
        self.replay_size = transition_params["replay_size"]
        self.bufferSize = transition_params["bufferSize"]

        self.recentMemSize = 1
        self.numEntries = 0
        self.insertIndex = 0
        self.buf_ind = self.bufferSize + 1 # To ensure the buffer is always refilled initially

        self.s = torch.zeros(self.replay_size, self.state_size).float()
        self.a = np.zeros((self.replay_size), dtype=np.int32)
        self.r = np.zeros((self.replay_size), dtype=np.float32)
        self.ret = np.zeros((self.replay_size), dtype=np.float32)
        self.ret_partial = np.zeros((self.replay_size), dtype=np.float32)
        self.t = np.zeros((self.replay_size), dtype=np.int32)
        self.steps_until_term = np.zeros((self.replay_size), dtype=np.int32)

        self.buf_a = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_r = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_ret = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_ret_partial = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_term = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_term_under_n = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_s = torch.zeros(self.bufferSize, self.state_size).float()
        self.buf_s2 = torch.zeros(self.bufferSize, self.state_size).float()

        if self.gpu >= 0:
            self.gpu_s  = self.buf_s.float().cuda()
            self.gpu_s2 = self.buf_s2.float().cuda()


    def size(self):
        return self.numEntries


    def fill_buffer(self):

        assert self.numEntries > self.bufferSize, 'Not enough transitions stored to learn'

        # clear CPU buffers
        self.buf_ind = 0

        for buf_ind in range(0, self.bufferSize):
            s, a, r, ret, ret_partial, s2, term, term_under_n = self.sample_one()
            self.buf_s[buf_ind].copy_(s)
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            self.buf_ret[buf_ind] = ret
            self.buf_ret_partial[buf_ind] = ret_partial
            self.buf_s2[buf_ind].copy_(s2)
            self.buf_term[buf_ind] = term
            self.buf_term_under_n[buf_ind] = term_under_n

        if self.gpu >= 0:
            self.gpu_s.copy_(self.buf_s)
            self.gpu_s2.copy_(self.buf_s2)


    def sample_one(self):

        assert self.numEntries > 1, 'Experience cache is empty'

        valid = False

        while not valid:
            index = randrange(1, self.numEntries - self.recentMemSize)
            if self.t[index + self.recentMemSize - 1] == 0:
                valid = True

        return self.get(index)


    def sample(self, batch_size):

        assert batch_size < self.bufferSize, 'Batch size must be less than the buffer size'

        if self.buf_ind + batch_size > self.bufferSize:
            self.fill_buffer()

        index = self.buf_ind
        index2 = index + batch_size

        self.buf_ind = self.buf_ind + batch_size

        if self.gpu >=0:
            return self.gpu_s[index:index2], self.buf_a[index:index2], self.buf_r[index:index2], self.buf_ret[index:index2], self.buf_ret_partial[index:index2], self.gpu_s2[index:index2], self.buf_term[index:index2], self.buf_term_under_n[index:index2]
        else:
            return self.buf_s[index:index2], self.buf_a[index:index2], self.buf_r[index:index2], self.buf_ret[index:index2], self.buf_ret_partial[index:index2], self.buf_s2[index:index2], self.buf_term[index:index2], self.buf_term_under_n[index:index2]


    def wrap_index(self, index):

        if self.numEntries == 0:
            return index

        while index < 0:
            index += self.numEntries

        while index >= self.numEntries:
            index -= self.numEntries

        return index


    def get(self, index):

        ar_index = index + self.recentMemSize - 1

        term_under_n = 0
        if (self.steps_until_term[ar_index] - 1) < self.n_step_n:
            term_under_n = 1

        return self.s[index], self.a[ar_index], self.r[ar_index], self.ret[ar_index], self.ret_partial[ar_index], self.s[index + 1], self.t[ar_index + 1], term_under_n


    def add(self, s, a, r, ret, ret_partial, term, steps_until_term):

        assert s is not None, 'State cannot be nil'
        assert a is not None, 'Action cannot be nil'
        assert r is not None, 'Reward cannot be nil'

        term_stored_value = 0
        if term:
            term_stored_value = 1

        # Overwrite (s, a, r, t) at insertIndex
        self.s[self.insertIndex] = s.clone()
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        self.ret[self.insertIndex] = ret
        self.ret_partial[self.insertIndex] = ret_partial
        self.t[self.insertIndex] = term_stored_value
        self.steps_until_term[self.insertIndex] = steps_until_term

        # Increment until at full capacity
        if self.numEntries < self.replay_size:
            self.numEntries += 1

        # Always insert at next index, then wrap around
        self.insertIndex += 1

        # Overwrite oldest experience once at capacity
        if self.insertIndex >= self.replay_size:
            self.insertIndex = 0
