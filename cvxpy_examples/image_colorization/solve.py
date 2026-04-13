import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

_dir = os.path.dirname(__file__)

img = mpimg.imread(os.path.join(_dir, "flower.png"))
img = img[:,:,0:3]
m, n, _ = img.shape

np.random.seed(5)
known_ind = np.where(np.random.rand(m, n) >= 0.90)
rows, cols = known_ind

M = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
R_known = img[:,:,0][known_ind]
G_known = img[:,:,1][known_ind]
B_known = img[:,:,2][known_ind]

def save_img(filename, R, G, B):
    out = np.stack((np.array(R), np.array(G), np.array(B)), axis=2)
    plt.imshow(out)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.)

R = cp.Variable((m, n))
G = cp.Variable((m, n))
B = cp.Variable((m, n))

constraints = [
    cp.abs(0.299*R + 0.587*G + 0.114*B - M) <= 1e-3,
    R >= 0, R <= 1,
    G >= 0, G <= 1,
    B >= 0, B <= 1,
    R[rows, cols] == R_known,
    G[rows, cols] == G_known,
    B[rows, cols] == B_known,
]

prob = cp.Problem(cp.Minimize(cp.tv(R, G, B)), constraints)
prob.solve(solver=cp.SCS, verbose=True)

print("Optimal TV value:", prob.value)
save_img(os.path.join(_dir, "flower_reconstructed.png"), R.value, G.value, B.value)
