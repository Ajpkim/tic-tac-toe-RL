from mdp import TicTacToeMDP
from Q_QF import TabularQ, NNQ, Q_learn, Q_learn_batch, make_NN
from train_save_play import *

policy_path = './policies/test'

if __name__ == '__main__':
    load_play_policy(policy_path)
