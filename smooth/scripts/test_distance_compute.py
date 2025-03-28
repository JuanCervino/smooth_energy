# Import packages.
import cvxpy as cp
import numpy as np

X0 = 15
X0d = 20

XT = 25
XTd = 0

# Generate a random non-trivial quadratic program.
T = 2
n = 3

A = np.array([[144/5,144/4*1/2,48/3*1/2],
              [144/4*1/2,36/3,24/2*1/2],
              [48/3*1/2,24/2*1/2,4]])


# P = np.array([[T**5, T**4, T**3, T**2, T],
#               [T**4, T**3, T**2, T**1, 1],
#               [T**3, T**2, T**1, T**0, 1],
#               [T**2, T**1, T**0, T**0, 1],
#               [T**1, T**0, T**0, T**0, 1]])

P = np.array([[T**5, T**4, T**3],
              [T**4, T**3, T**2],
              [T**3, T**2, T**1],])
A = A * P 




print("A", A.shape, A,np.linalg.det(A), np.linalg.eigvals(A))

p = np.array([[T**4, T**3, T**2]
              ,[4*T**3, 3*T**2, 2*T]])
b = np.array([X0d - X0 - T * XTd,
            XTd - XTd])

# Define and solve the CVXPY problem.
x = cp.Variable(n)

print(A.shape, p.shape, b.shape, x.shape)
prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) ),
                 [p @ x == b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)