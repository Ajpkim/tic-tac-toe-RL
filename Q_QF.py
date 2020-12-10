import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim_episode import sim_episode
from value_action_selection import *


# Q objs (TabularQ, NNQ) for storing and updating Q vals,
# and Q functions for training Q vals (Q_learn, Q_learn_batch)
# NNQ is not complete.

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
        if (s, a) not in self.q:  # initialize q val for unseen state action pairs
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


def Q_learn(mdp, Q, iters, n_episodes, lr=.1, eps=.3, verbose=False):
    """
    Q learning algorithm for updating Q vals with experience and current estimates of max q vals
    """
    print(f'Training Q agent with {iters * n_episodes} games...\n')

    if verbose:
        start = datetime.now()
        interval = iters // 10

    def pi(s, possible_actions) -> int:
        return epsilon_greedy(Q, s, possible_actions, eps)

    for i in range(iters):
        q_targets = []
        if verbose:
            if i % interval == 0:
                print(f'iter {i}:', datetime.now() - start)

        for j in range(n_episodes):
            episode = sim_episode(mdp, pi)
            for (s, a, r, s_prime) in episode:
                if s_prime is None:
                    target = r
                else:
                    # SHOULD THIS BE S_PRIME NOT S??? I THINK SO...
                    possible_actions = mdp.possible_actions_from_hash(s_prime)
                    target = r + mdp.discount_factor * \
                        value(Q, s_prime, possible_actions)
                q_targets += [(s, a, target)]

        Q.update(q_targets, lr)
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


class TicTacToeNN(nn.Module):
    "Network for approximating max q_val of action given state"

    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9, 27)
        self.fc2 = nn.Linear(27, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NNQ:
    """
    Doesn't work well.

    For storing and updating q values with NN's. 
    Each action has its own NN that is given states and targets for training.
    policy is model predictions.
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.actions = [i for i in range(9)]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.models = {a: TicTacToeNN().to(self.device)
                       for a in range(9)}  # init model for each possible action
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizers_loss = {a: (torch.optim.Adam(
            self.models[a].parameters(), lr=lr), nn.MSELoss()) for a in range(9)}
        # self.loss_fn = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(params=self.models[0].parameters(), lr=lr)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def get(self, s, a):
        with torch.no_grad():
            s = self.state2vec(s)
            s = torch.from_numpy(s).to(self.device).float()
            return self.models[a](s).item()  # return scalar

    def update(self, data, lr):
        # sort the data by action ... list of (state, target)'s for each action
        a_dict = {a: [] for a in self.actions}
        for (s, a, t) in data:
            a_dict[a].append((self.state2vec(s), t))

        # build training data for model
        for a in a_dict:
            if a_dict[a]:
                x_list, y_list = [list(tup) for tup in zip(*a_dict[a])]
                x, y = np.vstack(x_list), np.vstack(y_list)
                x, y = torch.from_numpy(x).to(self.device).float(
                ), torch.from_numpy(y).to(self.device).float()

                self.train(a, x, y)

    def state2vec(self, state):
        "Convert hash integer state to 1x9 row vector for NN"
        vec = np.zeros((1, 9))
        state = str(state)[::-1]
        for i in range(len(state)):
            vec[0, i] = float(state[i])  # np will convert to floats
        return vec

    ######## TRAINING LOOP ##########
    def train(self, a, x, y):
        "a is action in range(9), x is input, y is target"
        model = self.models[a]
        optimizer, loss_fn = self.optimizers_loss[a]
        # loss_fn = self.loss_fn
        # optimizer = self.optimizer

        optimizer.zero_grad()
        pred = model(x)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    pass
