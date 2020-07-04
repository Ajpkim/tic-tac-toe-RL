import random
from value_action_selection import *


def sim_episode(mdp, pi) -> list:
    """
    Simulate a game of tic-tac-toe given policy pi. 

    Args:
        - mdp: Deterministic Tic Tac Toe Markov Decision Process.
        - pi: policy function for determining action given state and possible moves based on a Q object.

    Return:
        - list episode history (state, action, reward, s_prime) for every step of game.
    """

    s = mdp.reset()  # initialize state

    # initialize list of state, action, reward tuples for episode (both players)
    all_s_a_r = []

    while True:  # play game
        # "player 1"
        poss_actions = mdp.get_possible_actions()
        a = pi(s, poss_actions)
        r, s2 = mdp.make_move(a, player_id=1)
        all_s_a_r.append((s, a, 0))  # 0 reward for every step except last

        if mdp.check_game_over():
            break

        # "player 2"
        poss_actions = mdp.get_possible_actions()
        a = pi(s2, poss_actions)
        r, s = mdp.make_move(a, player_id=2)

        all_s_a_r.append((s2, a, r))

        if mdp.check_game_over():
            break

    # split episode according to player_id to provide data from player POV.
    p1_s_a_r = all_s_a_r[::2]  # odd steps
    p2_s_a_r = all_s_a_r[1::2]  # even steps

    # assign rewards based on game outcome. Add reward to last step for each player.
    p1_reward = mdp.reward_fn(player_id=1)
    p1_s_a_r[-1] = p1_s_a_r[-1][0:2] + (p1_reward,)

    p2_reward = mdp.reward_fn(player_id=2)
    p2_s_a_r[-1] = p2_s_a_r[-1][0:2] + (p2_reward,)

    # create (s, a, r, s_prime) tuples for Q_learning
    p1_episode = [p1_s_a_r[i] + (p1_s_a_r[i+1][0],)
                  for i in range(len(p1_s_a_r)-1)]

    # player last move has no s_prime
    p1_episode.append(p1_s_a_r[-1] + (None,))

    p2_episode = [p2_s_a_r[i] + (p2_s_a_r[i+1][0],)
                  for i in range(len(p2_s_a_r)-1)]
    p2_episode.append(p2_s_a_r[-1] + (None,))

    episode = p1_episode + p2_episode

    return episode


if __name__ == '__main__':
    pass
