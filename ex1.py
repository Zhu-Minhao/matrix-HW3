import numpy as np


def power_method(A, x, max_iter=1000, tol=1e-6):

    x /= np.linalg.norm(x, np.inf)
    error = 1
    iterations = 0
    eigenvalue_estimate = []
    while error > tol and iterations < max_iter:
        y = A.dot(x)
        j = np.argmax(np.abs(y))
        if y[j] == 0:
            return 0, x
        eigenvalue_estimate = y[j]
        x_update = y / eigenvalue_estimate
        error = np.linalg.norm(x - x_update, np.inf)
        x = x_update
        iterations += 1
    return eigenvalue_estimate, x


A = np.random.rand(4, 4)
A = (A + A.T) / 2
x = np.random.rand(4)
a, v = power_method(A, x)
print("\nEigenvalue found by Power Method:", a)
print("Eigenvector found by Power Method:", v)
eigvals, eigvecs = np.linalg.eig(A)
principal_idx = np.argmax(np.abs(eigvals))
print("\nEigenvalue found by numpy.linalg.eig:", eigvals[principal_idx])
print("Eigenvector found by numpy.linalg.eig:", eigvecs[:, principal_idx])

