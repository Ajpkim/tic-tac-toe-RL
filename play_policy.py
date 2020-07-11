from mdp import TicTacToeMDP
from Q_QF import TabularQ, NNQ, Q_learn, Q_learn_batch
from train_save_play import *

policy_path = './policies/tabQ_5M'

if __name__ == '__main__':
    load_play_policy(policy_path, order=2)
