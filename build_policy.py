from mdp import TicTacToeMDP
from train_save_play import train_save_policy
from Q_QF import TabularQ, NNQ, Q_learn, Q_learn_batch, make_NN



# PARAMS
filename = './policies/tabQ_1M'
Q = TabularQ()
qf = Q_learn
episodes = 1000000
lr = .08
discount_factor = .9
eps = .3
rewards = {'win': 5, 'loss': -5, 'tie': -1}
verbose = True

if __name__ == '__main__':
    train_save_policy(filename=filename, Q=Q, qf=qf, episodes=episodes, lr=lr,
                      discount_factor=discount_factor, eps=eps, rewards=rewards, verbose=verbose)
