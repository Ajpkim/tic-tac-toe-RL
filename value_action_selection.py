import random


def value(Q, s, possible_actions):
    "Return estimated max q value for state s given possible actions in the mdp"
    return max((Q.get(s, a) for a in possible_actions))


def greedy(Q, s, possible_actions):
    "Return value maximizing action based on current q values. Return random maximizing action in case of tie."
    best = value(Q, s, possible_actions)
    return random.choice([a for a in possible_actions if Q.get(s, a) == best])


def epsilon_greedy(Q, s, possible_actions, eps=0.3):
    "Return random possible action with %eps and value maximizing action based on current q values rest of the time"
    if random.random() < eps:
        return random.choice(possible_actions)
    return greedy(Q, s, possible_actions)


def random_policy(Q, s, poss_actions):  # take 1 just so it matches other policy funcs
    "Return random action from given possible actions"
    return random.choice(poss_actions)


if __name__ == '__main__':
    pass
