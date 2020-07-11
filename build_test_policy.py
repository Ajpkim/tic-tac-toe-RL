from mdp import TicTacToeMDP
from Q_QF import TabularQ, NNQ, Q_learn, Q_learn_batch

Q = TabularQ()
mdp = TicTacToeMDP()

print(Q.q)
