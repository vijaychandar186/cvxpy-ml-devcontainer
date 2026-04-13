# CVXPY User Guide

---

## Table of Contents

1. [What is CVXPY?](#what-is-cvxpy)
2. [Changing the Problem](#changing-the-problem)
3. [Infeasible and Unbounded Problems](#infeasible-and-unbounded-problems)
4. [Other Problem Statuses](#other-problem-statuses)
5. [Vectors and Matrices](#vectors-and-matrices)
6. [Constraints](#constraints)
7. [Parameters](#parameters)
8. [Custom Labels](#custom-labels)
9. [Atomic Functions](#atomic-functions)
   - [Operators](#operators)
   - [Scalar Functions](#scalar-functions)
   - [Functions Along an Axis](#functions-along-an-axis)
   - [Elementwise Functions](#elementwise-functions)
   - [Vector/Matrix Functions](#vectormatrix-functions)
10. [Disciplined Convex Programming (DCP)](#disciplined-convex-programming-dcp)
11. [Disciplined Geometric Programming (DGP)](#disciplined-geometric-programming-dgp)
12. [Disciplined Parametrized Programming (DPP)](#disciplined-parametrized-programming-dpp)
13. [Disciplined Quasiconvex Programming (DQCP)](#disciplined-quasiconvex-programming-dqcp)
14. [Advanced Constraints](#advanced-constraints)
15. [Advanced Features](#advanced-features)
16. [Solver Features](#solver-features)

---

## What is CVXPY?

CVXPY is a Python-embedded modeling language for convex optimization problems. It automatically transforms the problem into standard form, calls a solver, and unpacks the results.

```python
import cvxpy as cp

# Create two scalar optimization variables.
x = cp.Variable()
y = cp.Variable()

# Create two constraints.
constraints = [x + y == 1,
               x - y >= 1]

# Form objective.
obj = cp.Minimize((x - y)**2)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
```

**Output:**
```
status: optimal
optimal value 0.999999999761
optimal var 1.00000000001 -1.19961841702e-11
```

The `status` field tells us the problem was solved successfully. The **optimal value** is the minimum value of the objective over all choices of variables that satisfy the constraints. `prob.solve()` returns the optimal value and updates `prob.status`, `prob.value`, and the `value` field of all variables in the problem.

---

## Changing the Problem

Problems are **immutable** — they cannot be changed after creation. To change the objective or constraints, create a new problem.

```python
# Replace the objective.
prob2 = cp.Problem(cp.Maximize(x + y), prob.constraints)
print("optimal value", prob2.solve())

# Replace the constraint (x + y == 1).
constraints = [x + y <= 3] + prob2.constraints[1:]
prob3 = cp.Problem(prob2.objective, constraints)
print("optimal value", prob3.solve())
```

**Output:**
```
optimal value 1.0
optimal value 3.00000000006
```

---

## Infeasible and Unbounded Problems

If a problem is infeasible or unbounded, the `status` field will be set to `"infeasible"` or `"unbounded"`, respectively. The `value` fields of the problem variables are not updated.

```python
import cvxpy as cp

x = cp.Variable()

# An infeasible problem.
prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)

# An unbounded problem.
prob = cp.Problem(cp.Minimize(x))
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
```

**Output:**
```
status: infeasible
optimal value inf
status: unbounded
optimal value -inf
```

> For a **minimization** problem: optimal value is `inf` if infeasible and `-inf` if unbounded. For **maximization** problems, the opposite is true.

---

## Other Problem Statuses

If the solver solves the problem to lower accuracy than desired, the problem status indicates the lower accuracy achieved:

| Status | Description |
|--------|-------------|
| `"optimal_inaccurate"` | Solved, but to lower accuracy |
| `"unbounded_inaccurate"` | Unbounded, but to lower accuracy |
| `"infeasible_inaccurate"` | Infeasible, but to lower accuracy |

CVXPY provides the following constants as aliases for status strings:

- `OPTIMAL`
- `INFEASIBLE`
- `UNBOUNDED`
- `OPTIMAL_INACCURATE`
- `INFEASIBLE_INACCURATE`
- `UNBOUNDED_INACCURATE`
- `INFEASIBLE_OR_UNBOUNDED`

To test if a problem was solved successfully:

```python
prob.status == OPTIMAL
```

> **Note:** `INFEASIBLE_OR_UNBOUNDED` is rare. You can determine the precise status by re-solving the problem with a constant objective (e.g., `cp.Minimize(0)`). If the new problem status is `INFEASIBLE_OR_UNBOUNDED`, the original was infeasible; if `OPTIMAL`, the original was unbounded.

---

## Vectors and Matrices

Variables can be scalars, vectors, or matrices (0, 1, or 2 dimensional).

```python
# A scalar variable.
a = cp.Variable()

# Vector variable with shape (5,).
x = cp.Variable(5)

# Column vector variable with shape (5, 1).
x = cp.Variable((5, 1))

# Matrix variable with shape (4, 7).
A = cp.Variable((4, 7))
```

Constants can be NumPy ndarrays or SciPy sparse matrices. Example:

```python
import cvxpy as cp
import numpy as np

m = 10
n = 5
numpy.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

print("Optimal objective value", prob.solve())
print("Optimal variable value")
print(x.value)
```

---

## Constraints

You can use `==`, `<=`, and `>=` to construct constraints. Equality and inequality constraints are **elementwise**, whether they involve scalars, vectors, or matrices.

> **Note:** You cannot use `<` or `>` (strict inequalities don't make sense in real-world optimization). Chained constraints like `0 <= x <= 1` are also not supported — CVXPY will raise an exception.

For **semidefinite cone constraints**, see the [Semidefinite Matrices](#semidefinite-matrices) section.

---

## Parameters

Parameters are symbolic representations of constants. They allow you to modify constant values without reconstructing the entire problem, enabling significant speedups via [DPP](#disciplined-parametrized-programming-dpp).

```python
# Positive scalar parameter.
m = cp.Parameter(nonneg=True)

# Column vector parameter with unknown sign (default).
c = cp.Parameter(5)

# Matrix parameter with negative entries.
G = cp.Parameter((4, 7), nonpos=True)

# Assigns a constant value to G.
G.value = -np.ones((4, 7))
```

You can initialize a parameter with a value — these are equivalent:

```python
# Create parameter, then assign value.
rho = cp.Parameter(nonneg=True)
rho.value = 2

# Initialize parameter with a value.
rho = cp.Parameter(nonneg=True, value=2)
```

### Trade-off Curves with Parameters

```python
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

n = 15
m = 10
np.random.seed(1)
A = np.random.randn(n, m)
b = np.random.randn(n)
gamma = cp.Parameter(nonneg=True)

x = cp.Variable(m)
error = cp.sum_squares(A @ x - b)
obj = cp.Minimize(error + gamma * cp.norm(x, 1))
prob = cp.Problem(obj)

sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = np.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    prob.solve()
    sq_penalty.append(error.value)
    l1_penalty.append(cp.norm(x, 1).value)
    x_values.append(x.value)
```

Trade-off curves can also be computed in **parallel**:

```python
from multiprocessing import Pool

def get_x(gamma_value):
    gamma.value = gamma_value
    result = prob.solve()
    return x.value

pool = Pool(processes=1)
x_values = pool.map(get_x, gamma_vals)
```

---

## Custom Labels

You can assign custom labels to expressions and constraints for easier debugging and model interpretation.

```python
import cvxpy as cp
import numpy as np

weights = cp.Variable(3, name="weights")

constraints = [
    (weights >= 0).set_label("non_negative_weights"),
    (cp.sum(weights) == 1).set_label("budget_constraint"),
    (weights <= 0.4).set_label("concentration_limits")
]

mu = np.array([0.08, 0.10, 0.07])
Sigma = np.array([[0.10, 0.02, 0.01],
                  [0.02, 0.08, 0.03],
                  [0.01, 0.03, 0.09]])

expected_return = (mu @ weights).set_label("expected_return")
risk = cp.quad_form(weights, Sigma).set_label("risk")

objective = cp.Minimize(risk - 0.5 * expected_return)
problem = cp.Problem(objective, constraints)

print(problem.format_labeled())
```

**Output:**
```
minimize risk + -0.5 @ expected_return
subject to non_negative_weights: 0.0 <= weights
           budget_constraint: Sum(weights, None, False) == 1.0
           concentration_limits: weights <= 0.4
```

Labels are "live" and can be modified after problem creation:

```python
risk.label = "portfolio_risk"
print(problem.format_labeled())
```

---

## Atomic Functions

### Operators

The infix operators `+`, `-`, `*`, `/`, `@` are treated as functions.

- `+` and `-` are always **affine**
- `expr1 @ expr2` — use for **matrix-matrix and matrix-vector multiplication**
- `expr1 * expr2` — use for **matrix-scalar and vector-scalar multiplication**
- Use `multiply` for elementwise multiplication

#### Indexing and Slicing

CVXPY follows NumPy semantics exactly:

```python
x = cvxpy.Variable(5)
print("0 dimensional", x[0].shape)    # ()
print("1 dimensional", x[0:1].shape)  # (1,)
```

Indexing **drops** dimensions; slicing **preserves** dimensions.

#### Transpose

```python
expr.T  # Transpose — affine function
```

#### Power

```python
expr**p  # Equivalent to power(expr, p)
```

---

### Scalar Functions

| Function | Meaning | Domain | Curvature |
|----------|---------|--------|-----------|
| `cvar(x, beta)` | Average of (1-β) fraction of largest values in x | x ∈ Rᵐ, β ∈ (0,1) | convex |
| `dotsort(X, W)` | Dot product of sort(vec(X)) and sort(vec(W)) | X ∈ Rᵐˣⁿ | convex |
| `geo_mean(x)` | x₁^(1/n) ⋯ xₙ^(1/n) | x ∈ Rⁿ₊ | concave |
| `harmonic_mean(x)` | n / (1/x₁ + ⋯ + 1/xₙ) | x ∈ Rⁿ₊ | concave |
| `inv_prod(x)` | (x₁⋯xₙ)⁻¹ | x ∈ Rⁿ₊ | convex |
| `lambda_max(X)` | λ_max(X) | X ∈ Sⁿ | convex |
| `lambda_min(X)` | λ_min(X) | X ∈ Sⁿ | concave |
| `lambda_sum_largest(X, k)` | Sum of k largest eigenvalues of X | X ∈ Sⁿ | convex |
| `lambda_sum_smallest(X, k)` | Sum of k smallest eigenvalues of X | X ∈ Sⁿ | concave |
| `log_det(X)` | log(det(X)) | X ∈ Sⁿ₊ | concave |
| `log_sum_exp(X)` | log(Σ eˣⁱʲ) | X ∈ Rᵐˣⁿ | convex |
| `matrix_frac(x, P)` | xᵀP⁻¹x | x ∈ Rⁿ, P ∈ Sⁿ₊₊ | convex |
| `max(X)` | max{Xᵢⱼ} | X ∈ Rᵐˣⁿ | convex |
| `mean(X)` | (1/mn) Σ Xᵢⱼ | X ∈ Rᵐˣⁿ | affine |
| `min(X)` | min{Xᵢⱼ} | X ∈ Rᵐˣⁿ | concave |
| `mixed_norm(X, p, q)` | Mixed norm | X ∈ Rⁿˣⁿ | convex |
| `norm(x)` / `norm(x, 2)` | √(Σ xᵢ²) | x ∈ Rⁿ | convex |
| `norm(x, 1)` | Σ |xᵢ| | x ∈ Rⁿ | convex |
| `norm(x, "inf")` | max{|xᵢ|} | x ∈ Rⁿ | convex |
| `norm(X, "fro")` | √(Σ Xᵢⱼ²) | X ∈ Rᵐˣⁿ | convex |
| `norm(X, "nuc")` | tr((XᵀX)^(1/2)) | X ∈ Rᵐˣⁿ | convex |
| `perspective(f(x), s)` | sf(x/s) | x ∈ dom(f), s ≥ 0 | same as f |
| `pnorm(X, p)` (p ≥ 1) | (Σ |Xᵢⱼ|ᵖ)^(1/p) | X ∈ Rᵐˣⁿ | convex |
| `pnorm(X, p)` (p < 1, p ≠ 0) | (Σ Xᵢⱼᵖ)^(1/p) | X ∈ Rᵐˣⁿ₊ | concave |
| `ptp(X)` | max Xᵢⱼ − min Xᵢⱼ | X ∈ Rᵐˣⁿ | convex |
| `quad_form(x, P)` (P ∈ Sⁿ₊) | xᵀPx | x ∈ Rⁿ | convex |
| `quad_form(x, P)` (P ∈ Sⁿ₋) | xᵀPx | x ∈ Rⁿ | concave |
| `quad_over_lin(X, y)` | (Σ Xᵢⱼ²)/y | x ∈ Rⁿ, y > 0 | convex |
| `sum(X)` | Σ Xᵢⱼ | X ∈ Rᵐˣⁿ | affine |
| `sum_largest(X, k)` | Sum of k largest Xᵢⱼ | X ∈ Rᵐˣⁿ | convex |
| `sum_smallest(X, k)` | Sum of k smallest Xᵢⱼ | X ∈ Rᵐˣⁿ | concave |
| `sum_squares(X)` | Σ Xᵢⱼ² | X ∈ Rᵐˣⁿ | convex |
| `trace(X)` | tr(X) | X ∈ Rⁿˣⁿ | affine |
| `tr_inv(X)` | tr(X⁻¹) | X ∈ Sⁿ₊₊ | convex |
| `tv(x)` | Σ |xᵢ₊₁ − xᵢ| | x ∈ Rⁿ | convex |
| `von_neumann_entr(X)` | −tr(X log(X)) | X ∈ Sⁿ₊ | concave |
| `std(X)` | Analog to numpy.std | X ∈ Rᵐˣⁿ | convex |
| `var(X)` | Analog to numpy.var | X ∈ Rᵐˣⁿ | convex |

> **Clarifications:**
> - `Sⁿ` = symmetric matrices; `Sⁿ₊` / `Sⁿ₋` = PSD / NSD; `Sⁿ₊₊` / `Sⁿ₋₋` = PD / ND
> - `norm(x)` and `norm(x, 2)` give the **Euclidean norm** for vectors, **spectral norm** for matrices
> - `max` and `min` give the largest/smallest entry; use `maximum`/`minimum` for lists of scalar expressions
> - `cp.sum` sums all entries in a single expression; Python's built-in `sum` adds a list of expressions

---

### Functions Along an Axis

`sum`, `norm`, `max`, `min`, `mean`, `std`, `var`, and `ptp` can be applied along an axis:

```python
X = cvxpy.Variable((5, 4))
col_sums = cvxpy.sum(X, axis=0, keepdims=True)  # shape (1, 4)
col_sums = cvxpy.sum(X, axis=0)                 # shape (4,)
row_sums = cvxpy.sum(X, axis=1)                 # shape (5,)
```

---

### Elementwise Functions

| Function | Meaning | Domain | Curvature |
|----------|---------|--------|-----------|
| `abs(x)` | |x| | x ∈ C | convex |
| `conj(x)` | Complex conjugate | x ∈ C | affine |
| `entr(x)` | −x log(x) | x > 0 | concave |
| `exp(x)` | eˣ | x ∈ R | convex |
| `huber(x, M=1)` | x² if |x| ≤ M, else 2M|x| − M² | x ∈ R | convex |
| `inv_pos(x)` | 1/x | x > 0 | convex |
| `kl_div(x, y)` | x log(x/y) − x + y | x > 0, y > 0 | convex |
| `log(x)` | log(x) | x > 0 | concave |
| `log_normcdf(x)` | Approx log of standard normal CDF | x ∈ R | concave |
| `log1p(x)` | log(x+1) | x > −1 | concave |
| `loggamma(x)` | Approx log of Gamma function | x > 0 | convex |
| `logistic(x)` | log(1 + eˣ) | x ∈ R | convex |
| `maximum(x, y)` | max{x, y} | x, y ∈ R | convex |
| `minimum(x, y)` | min{x, y} | x, y ∈ R | concave |
| `multiply(c, x)` | c·x | x ∈ R | affine |
| `neg(x)` | max{−x, 0} | x ∈ R | convex |
| `pos(x)` | max{x, 0} | x ∈ R | convex |
| `power(x, 0)` | 1 | x ∈ R | constant |
| `power(x, 1)` | x | x ∈ R | affine |
| `power(x, p)` (p = 2,4,8,…) | xᵖ | x ∈ R | convex |
| `power(x, p)` (p < 0) | xᵖ | x > 0 | convex |
| `power(x, p)` (0 < p < 1) | xᵖ | x ≥ 0 | concave |
| `power(x, p)` (p > 1, p ≠ 2,4,8,…) | xᵖ | x ≥ 0 | convex |
| `rel_entr(x, y)` | x log(x/y) | x > 0, y > 0 | convex |
| `scalene(x, alpha, beta)` | α·pos(x) + β·neg(x) | x ∈ R | convex |
| `sqrt(x)` | √x | x ≥ 0 | concave |
| `square(x)` | x² | x ∈ R | convex |
| `xexp(x)` | xeˣ | x ≥ 0 | convex |

> **Clarifications:** `log_normcdf` and `loggamma` are defined via approximations. `log_normcdf` has highest accuracy over [−4, 4].

---

### Vector/Matrix Functions

| Function | Meaning | Curvature |
|----------|---------|-----------|
| `bmat([[X11,…], …])` | Block matrix | affine |
| `convolve(c, x)` | c * x (convolution) | affine |
| `cumsum(X, axis=0)` | Cumulative sum along axis | affine |
| `diag(x)` | Diagonal matrix from vector | affine |
| `diag(X)` | Diagonal of matrix as vector | affine |
| `diff(X, k=1, axis=0)` | kth order differences along axis | affine |
| `hstack([X1, …, Xk])` | Horizontal stack | affine |
| `kron(X, Y)` | Kronecker product | affine |
| `outer(x, y)` | Outer product xyᵀ | affine |
| `partial_trace(X, dims, axis=0)` | Partial trace | affine |
| `partial_transpose(X, dims, axis=0)` | Partial transpose | affine |
| `reshape(X, (m', n'), order='F')` | Reshape | affine |
| `upper_tri(X)` | Flatten strictly upper-triangular part | affine |
| `vec(X)` | Vectorize (column-major) | affine |
| `vec_to_upper_tri(X, strict=False)` | Vector to upper triangular matrix | affine |
| `vstack([X1, …, Xk])` | Vertical stack | affine |

> **Clarifications:**
> - `bmat` input is a list of lists; elements in each inner list are stacked horizontally, then blocks are stacked vertically
> - `convolve(c, x)` output has size n+m−1, defined as yₖ = Σⱼ c[j]x[k−j]
> - `vec(X)` flattens X in column-major order: yᵢ = X_{i mod m, ⌊i/m⌋}
> - `reshape(X, (m', n'), order='F')` casts X into m'×n' in column-major order; `order='C'` uses row-major

---

## Disciplined Convex Programming (DCP)

DCP is a system for constructing mathematical expressions with known curvature from a library of base functions. CVXPY uses DCP to ensure optimization problems are convex. Visit [dcp.stanford.edu](https://dcp.stanford.edu) for an interactive introduction.

### Expressions

Expressions are formed from variables, parameters, numerical constants, arithmetic operators, and library functions.

```python
import cvxpy as cp

x, y = cp.Variable(), cp.Variable()
a, b = cp.Parameter(), cp.Parameter()

# Example expressions
3.69 + b/3
x - 4*a
sqrt(x) - minimum(y, x - a)
maximum(2.66 - sqrt(y), square(x + 2*y))
```

Dimensions are stored in `expr.shape`, total entries in `expr.size`, number of dimensions in `expr.ndim`.

### Sign

Each (sub)expression is flagged as **positive**, **negative**, **zero**, or **unknown**.

Rules for `expr1 * expr2`:
- **Zero** if either expression is zero
- **Positive** if both have the same known sign
- **Negative** if they have opposite known signs
- **Unknown** if either has unknown sign

```python
x = cp.Variable()
a = cp.Parameter(nonpos=True)
c = numpy.array([1, -1])

print("sign of x:", x.sign)            # UNKNOWN
print("sign of a:", a.sign)            # NONPOSITIVE
print("sign of square(x):", cp.square(x).sign)  # NONNEGATIVE
print("sign of c*a:", (c*a).sign)      # UNKNOWN
```

### Curvature

| Curvature | Meaning |
|-----------|---------|
| constant | f(x) independent of x |
| affine | f(θx + (1−θ)y) = θf(x) + (1−θ)f(y) |
| convex | f(θx + (1−θ)y) ≤ θf(x) + (1−θ)f(y) |
| concave | f(θx + (1−θ)y) ≥ θf(x) + (1−θ)f(y) |
| unknown | DCP analysis cannot determine curvature |

Any constant is also affine; any affine expression is both convex and concave.

```python
x = cp.Variable()
a = cp.Parameter(nonneg=True)

print("curvature of x:", x.curvature)           # AFFINE
print("curvature of a:", a.curvature)           # CONSTANT
print("curvature of square(x):", cp.square(x).curvature)  # CONVEX
print("curvature of sqrt(x):", cp.sqrt(x).curvature)      # CONCAVE
```

### Curvature Rules

`f(expr₁, expr₂, ..., exprₙ)` is **convex** if `f` is convex and for each `exprᵢ`:
- `f` is increasing in argument `i` and `exprᵢ` is convex, **or**
- `f` is decreasing in argument `i` and `exprᵢ` is concave, **or**
- `exprᵢ` is affine or constant

`f(expr₁, expr₂, ..., exprₙ)` is **concave** if `f` is concave and for each `exprᵢ`:
- `f` is increasing in argument `i` and `exprᵢ` is concave, **or**
- `f` is decreasing in argument `i` and `exprᵢ` is convex, **or**
- `exprᵢ` is affine or constant

`f(expr₁, expr₂, ..., exprₙ)` is **affine** if `f` is affine and each `exprᵢ` is affine.

### DCP Problems

Valid problem objectives:
- `Minimize(convex)`
- `Maximize(concave)`

Valid constraints:
- `affine == affine`
- `convex <= concave`
- `concave >= convex`

```python
x = cp.Variable()
y = cp.Variable()

# DCP problems
prob1 = cp.Problem(cp.Minimize(cp.square(x - y)), [x + y >= 0])
prob2 = cp.Problem(cp.Maximize(cp.sqrt(x - y)),
                   [2*x - 3 == y, cp.square(x) <= 2])

print("prob1 is DCP:", prob1.is_dcp())  # True
print("prob2 is DCP:", prob2.is_dcp())  # True

# Non-DCP problems
obj = cp.Maximize(cp.square(x))
prob3 = cp.Problem(obj)
print("prob3 is DCP:", prob3.is_dcp())  # False
```

> CVXPY raises an exception if you call `problem.solve()` on a non-DCP problem.

---

## Disciplined Geometric Programming (DGP)

DGP is an analog of DCP for **log-log convex functions** — functions of positive variables that are convex with respect to the geometric mean. DGP is a ruleset for log-log convex programs (LLCPs).

```python
import cvxpy as cp

# DGP requires Variables to be declared positive via `pos=True`.
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
z = cp.Variable(pos=True)

objective_fn = x * y * z
constraints = [4*x*y*z + 2*x*z <= 10, x <= 2*y, y <= 2*x, z >= 1]
problem = cp.Problem(cp.Maximize(objective_fn), constraints)
problem.solve(gp=True)  # Must pass gp=True
```

> **Note:** To solve DGP problems, you must pass `gp=True` to the `solve()` method.

### Log-log Curvature

A function f: D ⊆ Rⁿ₊₊ → R is **log-log convex** if F(u) = log f(eᵘ) is convex.

| Log-Log Curvature | Meaning |
|-------------------|---------|
| log-log constant | F is a constant (f is a positive constant) |
| log-log affine | F(θu + (1−θ)v) = θF(u) + (1−θ)F(v) |
| log-log convex | F(θu + (1−θ)v) ≤ θF(u) + (1−θ)F(v) |
| log-log concave | F(θu + (1−θ)v) ≥ θF(u) + (1−θ)F(v) |
| unknown | DGP analysis cannot determine curvature |

Every log-log affine function has the form: f(x) = c · x₁^a₁ · x₂^a₂ ⋯ xₙ^aₙ (a monomial).

### DGP Atoms — Scalar Functions

| Function | Meaning | Log-log Curvature |
|----------|---------|-------------------|
| `geo_mean(x)` | x₁^(1/n) ⋯ xₙ^(1/n) | log-log affine |
| `harmonic_mean(x)` | n / Σ(1/xᵢ) | log-log concave |
| `max(X)` | max{Xᵢⱼ} | log-log convex |
| `min(X)` | min{Xᵢⱼ} | log-log concave |
| `norm(x)` / `norm(x, 2)` | √(Σ xᵢ²) | log-log convex |
| `norm(X, "fro")` | √(Σ Xᵢⱼ²) | log-log convex |
| `norm(X, 1)` | Σ|Xᵢⱼ| | log-log convex |
| `norm(X, "inf")` | max|Xᵢⱼ| | log-log convex |
| `prod(X)` | Π Xᵢⱼ | log-log affine |
| `sum(X)` | Σ Xᵢⱼ | log-log convex |
| `sum_squares(X)` | Σ Xᵢⱼ² | log-log convex |
| `trace(X)` | tr(X) | log-log convex |
| `pf_eigenvalue(X)` | Spectral radius of X | log-log convex |

### DGP Atoms — Elementwise Functions

| Function | Meaning | Domain | Curvature |
|----------|---------|--------|-----------|
| `diff_pos(x, y)` | x − y | 0 < y < x | log-log concave |
| `entr(x)` | −x log(x) | 0 < x < 1 | log-log concave |
| `exp(x)` | eˣ | x > 0 | log-log convex |
| `log(x)` | log(x) | x > 1 | log-log concave |
| `maximum(x, y)` | max{x, y} | x, y > 0 | log-log convex |
| `minimum(x, y)` | min{x, y} | x, y > 0 | log-log concave |
| `multiply(x, y)` | x·y | x, y > 0 | log-log affine |
| `one_minus_pos(x)` | 1 − x | 0 < x < 1 | log-log concave |
| `power(x, p)` | xᵖ | x > 0 | log-log affine |
| `sqrt(x)` | √x | x > 0 | log-log affine |
| `square(x)` | x² | x > 0 | log-log affine |

### DGP Problems

Valid problem objectives:
- `Minimize(log-log convex)`
- `Maximize(log-log concave)`

Valid constraints:
- `log-log affine == log-log affine`
- `log-log convex <= log-log concave`
- `log-log concave >= log-log convex`

Check DGP compliance with `object.is_dgp()`.

---

## Disciplined Parametrized Programming (DPP)

> Requires CVXPY version ≥ 1.1.0

DPP is a ruleset for producing parametrized problems that CVXPY can re-canonicalize very quickly. The **first** solve compiles and caches the problem structure; **subsequent** solves reuse it — much faster.

### The DPP Ruleset (for DCP)

- All parameters are classified as **affine** (just like variables)
- The product of two expressions is affine when: one expression is **constant**, or one is **parameter-affine** and the other is **parameter-free**

Check DPP compliance:

```python
problem.is_dcp(dpp=True)
```

### Repeatedly Solving a DPP Problem

```python
import cvxpy as cp
import numpy

n = 15
m = 10
numpy.random.seed(1)
A = numpy.random.randn(n, m)
b = numpy.random.randn(n)
gamma = cp.Parameter(nonneg=True)

x = cp.Variable(m)
error = cp.sum_squares(A @ x - b)
obj = cp.Minimize(error + gamma * cp.norm(x, 1))
problem = cp.Problem(obj)
assert problem.is_dcp(dpp=True)

gamma_vals = numpy.logspace(-4, 1)
for val in gamma_vals:
    gamma.value = val
    problem.solve()  # Reuses compiled structure after first solve
```

### Sensitivity Analysis and Gradients

After solving with `requires_grad=True`, you can:
- Call `problem.backward()` → populates `param.gradient` with ∂(solution)/∂(param)
- Call `problem.derivative()` → populates `var.delta` with predicted change for a given `param.delta`

```python
x = cp.Variable()
p = cp.Parameter()
quadratic = cp.square(x - 2 * p)
problem = cp.Problem(cp.Minimize(quadratic))

p.value = 3.
problem.solve(requires_grad=True)
problem.backward()
print("The gradient is {0:0.1f}.".format(p.gradient))  # 2.0

p.delta = 1e-5
problem.derivative()
print("x.delta is {0:2.1g}.".format(x.delta))  # 2e-05
```

> **backward vs derivative:** Use `backward()` for gradients of a scalar-valued function of the solution w.r.t. parameters. Use `derivative()` for sensitivity analysis (how the solution changes given parameter perturbations) — much more efficient when there are multiple variables.

---

## Disciplined Quasiconvex Programming (DQCP)

DQCP is a generalization of DCP for **quasiconvex functions** (functions whose sublevel sets are convex). Every DCP problem is also DQCP.

```python
import cvxpy as cp

x = cp.Variable()
y = cp.Variable(pos=True)
objective_fn = -cp.sqrt(x) / y
problem = cp.Problem(cp.Minimize(objective_fn), [cp.exp(x) <= y])
problem.solve(qcp=True)  # Must pass qcp=True
```

### Curvature

DQCP adds two new curvature types: **quasiconvex** and **quasiconcave**. An expression that is both is **quasilinear**.

```python
x = cp.Variable(3)
y = cp.length(x)
z = -y
print(y.curvature)   # QUASICONVEX
print(z.curvature)   # QUASICONCAVE

w = cp.ceil(x)
print(w.curvature)   # QUASILINEAR
```

### DQCP Problems

Valid problem objectives:
- `Minimize(quasiconvex)`
- `Maximize(quasiconcave)`

Valid constraints:
- `affine == affine`
- `convex <= concave`
- `concave >= convex`
- `quasiconvex <= constant`
- `quasiconcave >= constant`

### DQCP Atoms

- **Ratio (`/`):** quasilinear when denominator has known sign
- **Scalar product (`*`):** quasiconcave when both args have same sign; quasiconvex when opposite
- **`dist_ratio(x, a, b)`:** ‖x−a‖₂/‖x−b‖₂ — quasiconvex
- **`gen_lambda_max(A, B)`:** Maximum generalized eigenvalue — quasiconvex
- **`condition_number(A)`:** λ_max(A)/λ_min(A) — quasiconvex
- **`ceil(x)`, `floor(x)`:** quasilinear
- **`sign(x)`:** quasilinear
- **`length(x)`:** index of last nonzero element — quasiconvex

### Solving DQCP Problems

```python
problem.solve(qcp=True)
# Optional bounds for bisection:
problem.solve(qcp=True, low=12, high=17)
# Use a different solver if subproblems fail:
problem.solve(qcp=True, solver=cp.SCS)
# Verbose output:
problem.solve(qcp=True, verbose=True)
```

---

## Advanced Constraints

### Attributes

Variables and parameters can be created with attributes:

```python
Leaf(shape=None, value=None, nonneg=False, nonpos=False, complex=False,
     imag=False, symmetric=False, diag=False, PSD=False, NSD=False,
     hermitian=False, boolean=False, integer=False, sparsity=None,
     pos=False, neg=False)
```

| Attribute | Description |
|-----------|-------------|
| `nonneg` | Constrained to be nonnegative |
| `nonpos` | Constrained to be nonpositive |
| `complex` | Complex-valued |
| `imag` | Purely imaginary |
| `symmetric` | Symmetric matrix |
| `diag` | Diagonal matrix |
| `PSD` | Symmetric positive semidefinite |
| `NSD` | Symmetric negative semidefinite |
| `hermitian` | Hermitian matrix |
| `boolean` | 0 or 1 values |
| `integer` | Integer values |
| `pos` | Positive |
| `neg` | Negative |
| `bounds` | Lower and upper bounds |

```python
p = Parameter(nonneg=True)
try:
    p.value = -1
except Exception as e:
    print(e)                      # Parameter value must be nonnegative.
print("Projection:", p.project(-1))  # 0.0
```

Use `leaf.value = leaf.project(val)` or `leaf.project_and_assign(val)` for safe value assignment.

> **Note:** Specifying attributes enables more fine-grained DCP analysis. However, dual variables are only recorded for **explicit constraints**, not for constraints defined via attributes.

### Sparsity Attribute

> Added in version 1.6

```python
# Upper triangular sparse variable
X = cp.Variable((10, 10), sparsity=np.triu_indices(n=10))

# Sparse variable based on condition on data
data = np.random.randn(10, 10)
X = cp.Variable((10, 10), sparsity=np.where(data > 0.5))

# Manual sparsity definition
X = cp.Variable((3, 3), sparsity=[(0, 1), (0, 2)])
```

Read/write sparse values via `.value_sparse` (a `scipy.sparse.coo_array`):

```python
sparsity = ([0, 1, 2, 2], [0, 2, 1, 2])
data = [1.3, 2.1, 0.7, 3.2]
P = cp.Parameter((3, 3), sparsity=sparsity)
P.value_sparse = coo_array((data, sparsity))
```

### Multiple Attributes

> Added in version 1.7

Certain combinations are supported, e.g., sparse integer variable:

```python
x = cp.Variable(shape=(2,2), sparsity=[(0,1),(0,1)], integer=True)
prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= -5.5])
prob.solve()
```

Bounded integer variable:

```python
x = cp.Variable(shape=(2,2), integer=True, bounds=[-1.5, 2])
```

> **Note:** Attributes that reduce dimensionality (`diag`, `symmetric`, `sparsity`) cannot be combined together.

### Semidefinite Matrices

**Method 1:** Use attribute `PSD=True`:

```python
X = cp.Variable((100, 100), PSD=True)
obj = cp.Minimize(cp.norm(X) + cp.sum(X))
```

**Method 2:** Use `>>` or `<<` operators:

```python
# expr1 must be positive semidefinite
constr1 = (expr1 >> 0)

# expr2 must be negative semidefinite
constr2 = (expr2 << 0)
```

`X >> Y` means zᵀ(X−Y)z ≥ 0 for all z ∈ Rⁿ. Both sides must be square matrices and affine.

Symmetry constraint:

```python
constr = (expr == expr.T)
```

### Mixed-Integer Programs

```python
# Boolean variable
x = cp.Variable(10, boolean=True)

# Integer variable
Z = cp.Variable((5, 7), integer=True)
```

**Supported open source MIP solvers:** HiGHS, GLPK_MI, CBC, SCIP

**Commercial solvers** (for large/challenging problems): CPLEX, GUROBI, XPRESS, MOSEK, COPT

### Boolean Logic Operations

```python
import cvxpy as cp

x = cp.Variable(3, boolean=True)
y = cp.Variable(3, boolean=True)

not_x = ~x              # NOT
both = x & y            # AND
either = x | y          # OR
exclusive = x ^ y       # XOR

# Functional syntax (3+ arguments)
any_of_three = cp.logic.Or(x, y, z)

# Implication and biconditional
x_implies_y = cp.logic.implies(x, y)
x_iff_y = cp.logic.iff(x, y)
```

### Complex Valued Expressions

```python
x = cp.Variable(complex=True)
p = cp.Parameter(imag=True)

print("p.is_imag() = ", p.is_imag())       # True
print("(x + 2).is_real() = ", (x + 2).is_real())  # False
```

Additional atoms for complex expressions:
- `real(expr)` — real part
- `imag(expr)` — imaginary part
- `conj(expr)` — complex conjugate
- `expr.H` — Hermitian (conjugate) transpose

---

## Advanced Features

### N-Dimensional Expressions

> Added in version 1.6 (experimental)

```python
x = cp.Variable((12, 10, 24))  # (locations, days, hours)

constraints = [
    cp.sum(x, axis=(0, 2)) <= 2000,  # daily usage across all locations
    x[:, :, :12] <= 100,              # first 12 hours at every location
    x[:, 3, :] == 0,                  # zero usage on the fourth day
]
```

### Dual Variables

```python
import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

constraints = [x + y == 1, x - y >= 1]
obj = cp.Minimize((x - y)**2)
prob = cp.Problem(obj, constraints)
prob.solve()

# Dual variable accessed via constraint.dual_value
print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
```

### Transforms

The `indicator` transform converts constraints into an expression (0 if constraints hold, ∞ if violated):

```python
x = cp.Variable()
constraints = [0 <= x, x <= 1]
expr = cp.transforms.indicator(constraints)
x.value = .5
print("expr.value = ", expr.value)  # 0.0
x.value = 2
print("expr.value = ", expr.value)  # inf
```

### Problem Arithmetic

```python
# Objective arithmetic
Minimize(expr1) + Minimize(expr2) == Minimize(expr1 + expr2)
Maximize(expr1) + Maximize(expr2) == Maximize(expr1 + expr2)
Minimize(expr1) - Maximize(expr2) == Minimize(expr1 - expr2)
alpha * Minimize(expr) == Minimize(alpha * expr)
-alpha * Minimize(expr) == Maximize(-alpha * expr)

# Problem arithmetic
prob1 + prob2 == Problem(prob1.objective + prob2.objective,
                         prob1.constraints + prob2.constraints)
alpha * prob == Problem(alpha * prob.objective, prob.constraints)
```

### Getting the Standard Form

```python
problem = cp.Problem(objective, constraints)
data, chain, inverse_data = problem.get_problem_data(cp.SCS)
soln = chain.solve_via_data(problem, data)
problem.unpack_results(soln, chain, inverse_data)
```

### Canonicalization Backends

| Backend | Description |
|---------|-------------|
| `CPP` (default) | Original C++ implementation (CVXCORE) |
| `SCIPY` | Pure Python using SciPy sparse — generally fast for vectorized problems |
| `COO` | Pure Python using 3D COO sparse tensors — best for DPP problems with large parameters |

```python
prob.solve(canon_backend=cp.SCIPY_CANON_BACKEND)
```

---

## Performance Tips

### 1. Vectorize Your Problem

**Bad (scalarized — slow):**
```python
constraints = []
for i in range(m):
    constraints.append(A[i, :] @ x == b[i])
```

**Good (vectorized — fast):**
```python
constraints = [A @ x == b]
```

For simple bounds, use the `bounds` attribute:
```python
x = cp.Variable(n, bounds=[0, 1])              # Best
constraints = [x >= 0, x <= 1]                 # Better (vectorized)
constraints = [x[i] >= 0 for i in range(n)]    # Slow
```

The `bounds` attribute supports scalars, NumPy arrays, parameters, and affine functions of parameters.

### 2. Use `cp.sum`, Not Python's `sum`

```python
# Slow: creates a deep binary tree of additions
objective = cp.Minimize(sum(cp.square(x)))

# Fast: single efficient operation
objective = cp.Minimize(cp.sum(cp.square(x)))
```

### 3. Use Parameters for Repeated Solves

```python
x = cp.Variable(n)
gamma = cp.Parameter(nonneg=True)
data = cp.Parameter(n)

prob = cp.Problem(cp.Minimize(cp.sum_squares(x - data) + gamma * cp.norm1(x)))
# First solve: compiles and caches
gamma.value = 0.1; data.value = np.random.randn(n); prob.solve()
# Subsequent solves: reuses compiled structure
gamma.value = 1.0; data.value = np.random.randn(n); prob.solve()
```

Verify DPP compliance with `prob.is_dpp()`.

### 4. Use `cp.sum_squares` for Quadratic Objectives

```python
# Slow and memory-intensive (constructs dense n×n identity matrix)
objective = cp.Minimize(cp.quad_form(x, np.eye(n)))

# Fast: purpose-built atom
objective = cp.Minimize(cp.sum_squares(x))
```

> Only use `cp.quad_form(x, P)` when P is a non-trivial PSD matrix. Sparse matrices are fine.

### 5. Choose the Right Canonicalization Backend

```python
prob.solve(canon_backend=cp.SCIPY_CANON_BACKEND)
```

### 6. Use Verbose Mode to Diagnose Slow Problems

```python
prob.solve(verbose=True)
```

### Summary

| Tip | Impact |
|-----|--------|
| Vectorize constraints and objectives | **Very high** — can reduce compile time by orders of magnitude |
| Use `cp.sum` instead of Python `sum` | **High** for large sums |
| Use parameters for repeated solves (DPP) | **High** — amortizes compile cost across solves |
| Use `cp.sum_squares` for quadratic objectives | **High** for large n |
| Choose the right canonicalization backend | **Moderate** — problem-dependent |
| Use `verbose=True` to find bottlenecks | **Diagnostic** — helps identify what to fix |

---

## Solver Features

### Solve Method Options

```python
prob.solve(
    solver=None,
    verbose=False,
    gp=False,
    qcp=False,
    requires_grad=False,
    enforce_dpp=False,
    ignore_dpp=False,
    **kwargs
)
```

| Argument | Description |
|----------|-------------|
| `solver` | Solver to use |
| `solver_path` | List of (solver, options) tuples to try in order |
| `verbose` | Show solver output |
| `gp` | Parse as DGP |
| `qcp` | Parse as DQCP |
| `requires_grad` | Enable backward/derivative gradient computation |
| `enforce_dpp` | Raise `DPPError` for non-DPP problems |
| `ignore_dpp` | Treat DPP problems as non-DPP (may speed up compilation) |

### Supported Solvers

| Solver | LP | QP | SOCP | SDP | EXP | POW | MIP |
|--------|----|----|------|-----|-----|-----|-----|
| CBC | ✓ | ✓ | | | | | ✓ |
| CLARABEL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| COPT | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓** |
| DAQP | ✓ | ✓ | | | | | |
| GLOP | ✓ | | | | | | |
| GLPK | ✓ | | | | | | |
| GLPK_MI | ✓ | | | | | | ✓ |
| OSQP | ✓ | ✓ | | | | | |
| PIQP | ✓ | ✓ | | | | | |
| PROXQP | ✓ | ✓ | | | | | |
| QPALM | ✓ | ✓ | | | | | |
| PDLP | ✓ | | | | | | |
| QOCO | ✓ | ✓ | ✓ | | | | |
| CPLEX | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| ECOS | ✓ | ✓ | ✓ | | ✓ | | |
| GUROBI | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| MOSEK | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓** |
| SCS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| SCIP | ✓ | ✓ | ✓ | | | | ✓ |
| XPRESS | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| HiGHS | ✓ | ✓ | ✓* | | | | ✓* |
| KNITRO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

*Mixed-integer LP only. **Except mixed-integer SDP.

Default solver selection: CLARABEL for SOCPs; OSQP for QPs; SCS as fallback for all (except MIP).

```python
# Solve with a specific solver
prob.solve(solver=cp.OSQP)
prob.solve(solver=cp.CLARABEL)

# List all installed solvers
print(installed_solvers())
```

### Solver Stats

```python
prob.solve()
print(prob.solver_stats.solve_time)
```

### Warm Start

```python
prob.solve()                      # First solve
prob.solve(warm_start=True)       # Faster subsequent solve (reuses factorization)
```

Warm start is enabled by default. The initial guess for the first solve is constructed from variable `.value` fields.

### Solver Options

#### OSQP
| Option | Default |
|--------|---------|
| `max_iter` | 10,000 |
| `eps_abs` | 1e-5 |
| `eps_rel` | 1e-5 |

#### ECOS
| Option | Default |
|--------|---------|
| `max_iters` | 100 |
| `abstol` | 1e-8 |
| `reltol` | 1e-8 |
| `feastol` | 1e-8 |

#### SCS
| Option | Default |
|--------|---------|
| `max_iters` | 2500 |
| `eps` | 1e-4 |
| `alpha` | 1.8 |
| `acceleration_lookback` | 10 |
| `use_indirect` | False |

#### MOSEK
```python
prob.solve(solver=cp.MOSEK,
           mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'})
```

#### CLARABEL
| Option | Default |
|--------|---------|
| `max_iter` | 50 |
| `time_limit` | 0.0 (no limit) |

#### SCS (verbose example)
```python
prob.solve(solver=cp.SCS, verbose=True, use_indirect=True)
```

#### CBC Cut Generators
Available: `GomoryCuts`, `MIRCuts`, `MIRCuts2`, `TwoMIRCuts`, `ResidualCapacityCuts`, `KnapsackCuts`, `FlowCoverCuts`, `CliqueCuts`, `LiftProjectCuts`, `AllDifferentCuts`, `OddHoleCuts`, `RedSplitCuts`, `LandPCuts`, `PreProcessCuts`, `ProbingCuts`, `SimpleRoundingCuts`

```python
prob.solve(solver=cp.CBC, GomoryCuts=True, maximumSeconds=60)
```

#### HiGHS
```python
prob.solve(solver=cp.HIGHS, presolve="off")
# Or via dictionary:
prob.solve(solver=cp.HIGHS, highs_options=dict(solver="simplex"))
```

### Custom Solvers

```python
import cvxpy as cp
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP


class CUSTOM_OSQP(OSQP):
    MIP_CAPABLE = False

    def name(self):
        return "CUSTOM_OSQP"

    def solve_via_data(self, *args, **kwargs):
        print("Solving with a custom QP solver!")
        super().solve_via_data(*args, **kwargs)


x = cp.Variable()
quadratic = cp.square(x)
problem = cp.Problem(cp.Minimize(quadratic))
problem.solve(solver=CUSTOM_OSQP())
```

Inherit from either:
- `cvxpy.reductions.solvers.qp_solvers.qp_solver.QpSolver`
- `cvxpy.reductions.solvers.conic_solvers.conic_solver.ConicSolver`

> The string returned by `name()` must be different from all officially supported solvers (see `cvxpy.settings.SOLVERS`). Set `MIP_CAPABLE = True` and `MI_SUPPORTED_CONSTRAINTS` if your solver supports mixed-integer programs.

---