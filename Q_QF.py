import numpy as np
from datetime import datetime

from sim_episode import sim_episode
from value_action_selection import *

# Q objs (TabularQ, NNQ) for storing and updating Q vals,
# and Q functions for training Q vals (Q_learn, Q_learn_batch)


class TabularQ():
    """
    For storing and updating q values with a dictionary.
    Keys are tuples of hashed game states and action.
    """

    def __init__(self):
        self.q = {}

    def set(self, s, a, v):
        self.q[(s, a)] = v

    def get(self, s, a):
        if (s, a) not in self.q:
            self.q[(s, a)] = 0
        return self.q[(s, a)]

    def update(self, experiences, lr):
        """
        Q update step based on Q learning algo:
            q_new <- q_old + learning_rate * (reward + discount_factor * max_Q(s_prime, a) - q_old)

        "target" is current estimate of optimal future value: reward + disc. * max_Q(s_prime, a)
        """
        for (s, a, target) in experiences:
            if (s, a) not in self.q:
                self.q[(s, a)] = 0
            self.q[(s, a)] = (1 - lr) * self.q[(s, a)] + lr * target


class NNQ():
    """
    For storing and updating q values with NN's. 
    Each action has its own NN that is given states and targets for training.
    policy is model predictions.
    """
    def __init__():
        self.models = {}


def make_NN():
    # model =
    pass


def Q_learn(mdp, Q, episodes, lr=.1, eps=.3, verbose=False):
    "Q learning algorithm for learning q values one datum at a time"

    print('Training Q agent with {} episodes...\n'.format(episodes))

    if verbose:
        start = datetime.now()
        interval = episodes // 10

    def pi(s, possible_actions) -> int:
        return epsilon_greedy(Q, s, possible_actions, eps)

    for i in range(episodes):
        if verbose:
            if i % interval == 0:
                print(f'episode {i}:', datetime.now() - start)

        episode = sim_episode(mdp, pi)
        for (s, a, r, s_prime) in episode:
            if s_prime is None:
                target = r
            else:
                possible_actions = mdp.possible_actions_from_hash(s)
                target = r + mdp.discount_factor * \
                    value(Q, s_prime, possible_actions)

            Q.update([(s, a, target)], lr)

    return Q


def Q_learn_batch(mdp, Q, episodes, lr=.1, eps=.3, verbose=False):
    """Q learning algo for learning q vlaues. Stores all experience and adds new experiences to train on in batches"""

    if verbose:
        interval = episodes // 10
        start = datetime.now()

    print(f'Bacth training Q agent with {episodes} episodes....\n')

    def pi(s, possible_actions) -> int:
        return epsilon_greedy(Q, s, possible_actions, eps)

    all_experiences = []
    for i in range(episodes):
        if verbose:
            if i % interval == 0:
                print(f'episode {i}:', datetime.now() - start)

        episode = sim_episode(mdp, pi)
        all_experiences += episode

        # Update all q vals each episode given new data
        all_q_targets = []

        for (s, a, r, s_prime) in all_experiences:
            if s_prime is None:
                # No future expected value, only terminal reward resulting from (s,a)
                target = r
            else:
                # consider the possible actions given context of game
                poss_actions = mdp.possible_actions_from_hash(s)
                # r is 0 in our case for every action except terminal when target is set to 0 anyway
                target = r + mdp.discount_factor * \
                    value(Q, s_prime, poss_actions)

            all_q_targets.append((s, a, target))

        Q.update(all_q_targets, lr)

    return Q


if __name__ == '__main__':
    pass
