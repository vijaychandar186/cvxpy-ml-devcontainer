import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data import train, test

a_hat = np.load(os.path.join(os.path.dirname(__file__), "a_hat.npy"))

j = test[:, 0] - 1

k = test[:, 1] - 1

y_actual = test[:, 2]

ml_acc = np.mean(np.sign(a_hat[j] - a_hat[k]) == y_actual)

bl_acc = np.mean(train[:, 2] == y_actual)

print("ml accuracy", ml_acc)

print("baseline", bl_acc)
