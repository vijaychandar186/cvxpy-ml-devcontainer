import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cvxpy as cp
from scipy.sparse import coo_matrix
from data import n, m, sigma, train

rows = np.arange(m)
A1 = coo_matrix(
(train[:, 2], (rows, train[:, 0].astype(int)-1)),
shape=(m, n)
)
A2 = coo_matrix(
(-train[:, 2], (rows, train[:, 1].astype(int)-1)),
shape=(m, n)
)
A = A1 + A2

a = cp.Variable(n)

prob = cp.Problem(cp.Maximize(cp.sum(-cp.logistic(-2 * (A @ a) / sigma))), [a >= 0, a <= 1])

prob.solve(verbose=True)

a_hat = a.value
np.save(os.path.join(os.path.dirname(__file__), "a_hat.npy"), a_hat)
