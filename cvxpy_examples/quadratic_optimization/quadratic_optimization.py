import cvxpy as cp
import numpy as np

x = cp.Variable(2)

# x1^2 + 2x2^2 - x1x2 - x1 as quad_form
Q = np.array([[1, -0.5], [-0.5, 2]])
c = np.array([-1, 0])
obj = cp.Minimize(cp.quad_form(x, Q) + c @ x)

u1, u2 = -2, -3
A = np.array([[1, 2], [1, -4], [5, 76]])
b = np.array([u1, u2, 1])

constraints = [A @ x <= b]
prob = cp.Problem(obj, constraints)
prob.solve(verbose=True)

print("x1:", x.value[0], "x2:", x.value[1])
print("p*:", prob.value)
print("duals:", constraints[0].dual_value)

lam = constraints[0].dual_value
p_star = prob.value

print("\ndelta1  delta2  p*_pred    p*_exact")
for d1 in [-0.1, 0, 0.1]:
    for d2 in [-0.1, 0, 0.1]:
        delta = np.array([d1, d2, 0])
        p_pred = p_star - lam @ delta

        b_new = np.array([u1+d1, u2+d2, 1])
        prob_e = cp.Problem(obj, [A @ x <= b_new])
        prob_e.solve()

        print(f"  {d1:+.1f}    {d2:+.1f}    {p_pred:.4f}    {prob_e.value:.4f}")
