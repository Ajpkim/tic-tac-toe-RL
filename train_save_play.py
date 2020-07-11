import pickle

from mdp import TicTacToeMDP
from Q_QF import TabularQ, NNQ, Q_learn, Q_learn_batch
from value_action_selection import *

# Functions to train, save, load, and play against Q agents.


def train_save_policy(filename, Q, qf, iters, n_episodes, lr, discount_factor, eps, rewards, verbose=False):
    mdp = TicTacToeMDP(rewards, discount_factor)
    Q = qf(mdp, Q, iters, n_episodes, lr, eps, verbose)  # TRAIN

    with open(filename, mode='wb') as file:
        pickle.dump(Q, file)

    print('\ndumped policy at', filename)

    return Q


def load_play_policy(filename, order):

    with open(filename, mode='rb') as file:
        policy = pickle.load(file)

    play_human(policy, order)


def play_human(Q, order=1):
    mdp = TicTacToeMDP()
    mdp.print_board()
    while True:

        # Q GOES FIRST
        if order == 1:
            s = mdp.get_hashed_state()
            poss_actions = mdp.get_possible_actions()
            a = greedy(Q, s, poss_actions)
            print()
            print('State:', s)
            print('Agent selects:', a)
            mdp.make_move(a, player_id=1)
            mdp.print_board()

            if mdp.check_game_over():
                print()
                mdp.print_board()
                print('Game over')
                return

            # human player
            print()
            poss_actions = mdp.get_possible_actions()
            a = int(input('select an action: '))
            while a not in poss_actions:
                print('illegal action. Try again.')
                a = int(input('select an action: \n'))

            mdp.make_move(a, player_id=2)
            print()
            mdp.print_board()

            if mdp.check_game_over():
                print()
                mdp.print_board()
                print('Game over')
                return

        # HUMAN GOES FIRST
        else:
            print()
            poss_actions = mdp.get_possible_actions()
            a = int(input('select an action: '))
            while a not in poss_actions:
                print('illegal action. Try again.')
                a = int(input('select an action: \n'))

            mdp.make_move(a, player_id=1)
            print()
            mdp.print_board()

            if mdp.check_game_over():
                print()
                mdp.print_board()
                print('Game over')
                return

            s = mdp.get_hashed_state()
            poss_actions = mdp.get_possible_actions()
            a = greedy(Q, s, poss_actions)
            print()
            print('State:', s)
            print('Agent selects:', a)
            mdp.make_move(a, player_id=2)
            mdp.print_board()

            if mdp.check_game_over():
                print()
                mdp.print_board()
                print('Game over')
                return


if __name__ == '__main__':
    pass
