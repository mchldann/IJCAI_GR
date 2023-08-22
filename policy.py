import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from typing import Tuple
from torch.distributions import Categorical
from random import randrange


class Policy(object):

    @abstractmethod
    def sample_action(self, q_values) -> Tuple[int, float]:
        pass


class SoftmaxPolicy(Policy):

    def __init__(self, temperature):
        self.temperature = temperature

    def sample_action(self, q_values) -> int:
        bestq = q_values.max().item()
        probs = F.softmax(q_values.div(self.temperature), dim=0)
        prob_dist = Categorical(probs)
        return (prob_dist.sample().item(), bestq)


class GreedyPolicy(Policy):

    def sample_action(self, q_values) -> int:
        q = q_values.numpy()
        maxq = q[0]
        besta = [0]

        # Evaluate all other actions (with random tie-breaking)
        for a in range(1, q.shape[0]):

            if q[a] > maxq:
                besta = [a]
                maxq = q[a]

            elif q[a] == maxq:
                besta.append(a)

        r = randrange(len(besta))
        action_selected = besta[r]

        bestq = q[action_selected]

        return (action_selected, bestq)


class eGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.greedy_policy = GreedyPolicy()

    def sample_action(self, q_values) -> int:
        action, bestq = self.greedy_policy.sample_action(q_values)

        # Epsilon greedy
        if np.random.uniform() < self.epsilon:
            return (randrange(q_values.size()[0]), bestq)
        else:
            return (action, bestq)
