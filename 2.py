import numpy as np


# def generate_non_singular_matrix(n):
#     while True:
#         A = np.random.rand(n, n)
#         if np.linalg.det(A) != 0:
#             return A


def gram_schmidt(A):

    n = A.shape[0]
    Q = np.zeros((n, n))
    R = np.zeros((n, n))

    for k in range(n):
        Q[:, k] = A[:, k]
        for i in range(k):
            R[i, k] = np.dot(Q[:, i], A[:, k])
            Q[:, k] -= R[i, k] * Q[:, i]
        Q[:, k] /= np.linalg.norm(Q[:, k])
        R[k, k] = np.dot(Q[:, k], A[:, k])

    return Q, R


A = generate_non_singular_matrix(4)

Q, R = gram_schmidt(A)
Q_times_R = np.dot(Q, R)
Q_times_Q_transpose = np.dot(Q, np.transpose(Q))
print("Generated non-singular matrix A:\n", A)
print("Q:\n", Q)
print("R:\n", R)
print("Q_times_R:\n", Q_times_R)
print("Q_times_Q_transpose:\n", Q_times_Q_transpose)