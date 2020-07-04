import pickle

from mdp import TicTacToeMDP
from Q_QF import TabularQ, NNQ, Q_learn, Q_learn_batch, make_NN
from value_action_selection import *


def train_save_policy(filename, Q, qf, episodes, lr, discount_factor, eps, rewards, verbose=False):
    mdp = TicTacToeMDP(rewards, discount_factor)
    Q = qf(mdp, Q, episodes, lr, eps, verbose)

    with open(filename, mode='wb') as file:
        pickle.dump(Q, file)

    print('\ndumped policy at', filename)

    return Q


def load_play_policy(filename):

    with open(filename, mode='rb') as file:
        policy = pickle.load(file)

    play_human(policy)


def play_human(Q):
    mdp = TicTacToeMDP()
    mdp.print_board()
    while True:
        s = mdp.get_hashed_state()
        poss_actions = mdp.get_possible_actions()
        a = greedy(Q, s, poss_actions)
        print('\nAgent selects:', a)
        mdp.make_move(a, player_id=1)
        mdp.print_board()

        if mdp.check_game_over():
            mdp.print_board()
            print('Game over')
            return

        # human player
        print()
        a = int(input('select an action: '))
        mdp.make_move(a, player_id=2)
        print()
        mdp.print_board()

        if mdp.check_game_over():
            mdp.print_board()
            print('Game over')
            return


if __name__ == '__main__':
    pass
