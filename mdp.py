import numpy as np


class TicTacToeMDP():
    """
    Deterministic Markov Decision Process for simulating games of tic tac toe.
    MDP keeps represents state as row vector. Agents deal with correspodning integer states.
        i.e. internal vector state [2,1,0,0,0,2,1,1,2] = 211200012 as an integer state.

    Args:
        - rewards: dictionary with keys 'win', 'loss', 'tie' and float vals
        - discount_factor: float
    """

    def __init__(self, rewards={'win': 1, 'loss': -1, 'tie': 0}, discount_factor=0.9):
        self.rewards = rewards
        self.state = np.zeros((1, 9))
        self.hashed_state = 0
        self.discount_factor = discount_factor

    def reset(self) -> np.array:
        "Reset and return hashed blank game state"
        self.state = np.zeros((1, 9))
        return self.get_hashed_state()

    def get_possible_actions(self) -> list:
        "Return list of indices of empty spaces (representing possible actions) in current game state"
        return [i for i in range(9) if self.state[0, i] == 0]

    def possible_actions_from_hash(self, hash: int) -> list:
        "Return list of possible actions from state represented by hash"
        state = self.hash_to_vec(hash)
        return [i for i in range(9) if state[0, i] == 0]

    def make_move(self, a: int, player_id: int) -> tuple:
        "Update game state with player id(1 or 2) at location a. Return reward and hashed state."
        self.state[0, a] = player_id  # update internal board state
        reward = self.reward_fn(player_id)
        s_prime = self.get_hashed_state()  # return to agent the hashed board state
        return reward, s_prime

    def reward_fn(self, player_id: int) -> float:
        "return reward for player given current state"

        status = self.check_game_over()

        if status == 0:
            return 0.0  # game NOT over
        if status == -1:
            return self.rewards['tie']  # tie

        elif status == 1:  # p1 won
            if player_id == 1:
                return self.rewards['win']
            else:
                return self.rewards['loss']

        elif status == 2:  # p2 won
            if player_id == 2:
                return self.rewards['win']
            else:
                return self.rewards['loss']

    def get_hashed_state(self) -> int:
        "Convert current state (1x9 row vector) to int"
        return int(sum([self.state[0, i] * 10**i for i in range(9)]))

    def hash_to_vec(self, hash: int) -> np.array:
        "Convert integer state to vector state"
        vec = np.zeros((1, 9))
        s = str(hash)[::-1]
        for i in range(len(s)):
            vec[0, i] = (s[i])  # np will convert to floats
        return vec

    def check_game_over(self) -> int:
        """
        Return status of game:
            not over -> 0
            p1 win -> 1
            p2 win -> 2
            tie    -> -1
        """
        board = self.state.reshape(3, 3)
        p1_win = np.array([1, 1, 1])
        p2_win = np.array([2, 2, 2])

        # check diagonals
        if 1 == board[0, 0] == board[1, 1] == board[2, 2] or 1 == board[0, 2] == board[1, 1] == board[2, 0]:
            return 1
        if 2 == board[0, 0] == board[1, 1] == board[2, 2] or 2 == board[0, 2] == board[1, 1] == board[2, 0]:
            return 2

        # check cols and rows
        for i in range(3):
            if np.array_equal(board[i], p1_win) or np.array_equal(board.T[i], p1_win):
                return 1
            if np.array_equal(board[i], p2_win) or np.array_equal(board.T[i], p2_win):
                return 2

        # tie condition
        if 0 not in board[:, :]:
            return -1

        # game not over
        return 0

    def print_board(self):
        print(self.state.reshape(3, 3))

    def print_board_hash(self, hash: int):
        "Print board represented by hash state"
        state = self.hash_to_vec(hash)
        print(state.reshape(3, 3))


if __name__ == '__main__':
    pass
