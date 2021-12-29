import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from globals import *
from Matrix_A import Matrix_A
from Matrix_b import Matrix_b

def JacobiMethod(A, b, N, maxResiduum, write):
    if write == True:
        print("Jacobi Method, N = " + str(N))

    t0 = time.time()

    r = np.ones((N, 1))
    L = np.tril(A.getMatrix_A(), -1)
    U = np.triu(A.getMatrix_A(), 1)
    D = np.diag(np.diag(A.getMatrix_A()))

    LU = L + U
    D_ld_LU = np.linalg.solve(D, LU)
    D_ld_b = np.linalg.solve(D, b)

    iteration = 0

    while True:
        # r = -D\(L + U) * r + D\b - Jacobi matrix form
        r = - D_ld_LU.dot(r) + D_ld_b

        res = A.getMatrix_A().dot(r) - b

        normRes = np.linalg.norm(res)
        # it ends when the norm has exceeded the maxResiduum or NaN
        # or when the loop has not been completed for 500 iterations
        if normRes <= maxResiduum or np.isnan(normRes) or iteration >= 500:
            break

        iteration = iteration + 1


    t1 = time.time()
    total_time = t1 - t0

    if write == True:
        print("Iteration = " + str(iteration))
        print("Residuum norm = " + str(normRes))
        print("Time = " + str(total_time) + "s\n")

    return total_time

def GaussSeidlMethod(A, b, N, maxResiduum, write):
    if write == True:
        print("Gauss-Seidl Method, N = " + str(N))

    t0 = time.time()

    r = np.ones((N, 1))
    L = np.tril(A.getMatrix_A(), -1)
    U = np.triu(A.getMatrix_A(), 1)
    D = np.diag(np.diag(A.getMatrix_A()))

    DL = D+L
    # ld - left division matrix
    DL_ld_b = np.linalg.solve(DL, b)

    DL_ld_U = np.linalg.solve(DL, U)

    iteration = 0

    while True:
        # r = -(D+L)\(U*r)+(D+L)\b - Gauss-Seidl matrix form
        r = - DL_ld_U.dot(r) + DL_ld_b

        res = A.getMatrix_A().dot(r) - b

        normRes = np.linalg.norm(res)
        # it ends when the norm has exceeded the maxResiduum or NaN
        # or when the loop has not been completed for 500 iterations
        if normRes <= maxResiduum or np.isnan(normRes) or iteration >= 500:
            break

        iteration = iteration + 1


    t1 = time.time()
    total_time = t1 - t0

    if write == True:
        print("Iteration = " + str(iteration))
        print("Residuum norm = " + str(normRes))
        print("Time = " + str(total_time) + "s\n")

    return total_time

def LU_decompositionMethod(A, b, N, write):
    if write == True:
        print("LU Decomposition Method, N = " + str(N))

    t0 = time.time()

    L = np.zeros((N, N))
    np.fill_diagonal(L, 1)
    U = A.getMatrix_A().copy()

    # getting L and U
    for k in range(N - 1):
        for j in range(k + 1, N):
            L[j][k] = U[j][k] / U[k][k]
            U[j][k:N] = U[j][k:N] - L[j][k] * U[k][k:N]

    y = np.zeros((N, 1))
    x = np.zeros((N, 1))

    #  L * y = b using forward substitution method
    for i in range(N):
        Sum = 0.0
        for j in range(i):
            Sum += L[i][j] * y[j][0]
        y[i][0] = (b[i][0] - Sum) / L[i][i]

    #  U * x = y using backward substitution method
    for i in range(N - 1, -1, -1):
        Sum = 0.0
        for j in range(N - 1, i, -1):
            Sum += U[i][j] * x[j][0]
        x[i][0] = (y[i][0] - Sum) / U[i][i]


    res = A.getMatrix_A().dot(x) - b
    normRes = np.linalg.norm(res)

    t1 = time.time()
    total_time = t1 - t0

    if write == True:
        print("Residuum norm = " + str(normRes))
        print("Time = " + str(total_time) + "s\n")

    return total_time




A_A = Matrix_A(a1_A, N_A)
b = Matrix_b(N_A)
A_C = Matrix_A(a1_C, N_A)
maxResiduum = pow(10, -9)

timeJacobi = []
timeGaussSeidl = []
timeLU = []

print("Task B\n")
JacobiMethod(A_A, b.getMatrix_b(), N_A, maxResiduum, True)
GaussSeidlMethod(A_A, b.getMatrix_b(), N_A, maxResiduum, True)
print("Task C\n")
JacobiMethod(A_C, b.getMatrix_b(), N_A, maxResiduum, True)
GaussSeidlMethod(A_C, b.getMatrix_b(), N_A, maxResiduum, True)
print("Task D\n")
LU_decompositionMethod(A_C, b.getMatrix_b(), N_A, True)
print("Task E\n")
for i in range(len(N_E)):
    A_E = Matrix_A(a1_A, N_E[i])
    b = Matrix_b(N_E[i])
    timeJacobi.append(JacobiMethod(A_E, b.getMatrix_b(), N_E[i], maxResiduum, False))
    timeGaussSeidl.append(GaussSeidlMethod(A_E, b.getMatrix_b(), N_E[i], maxResiduum, False))
    timeLU.append(LU_decompositionMethod(A_E, b.getMatrix_b(), N_E[i], False))
    print("All methods for N = " + str(N_E[i]) + " were performed.")

print("")
print("N sizes = " + str(N_E))
print("Time Jacobi = " + str(timeJacobi))
print("Time GaussSeidl = " + str(timeGaussSeidl))
print("Time LU = " + str(timeLU))

figure(num=None, figsize=(20, 9), dpi=60)
plt.plot(N_E, timeJacobi, label="Jacobi")
plt.plot(N_E, timeGaussSeidl, label="Gauss-Seidl")
plt.plot(N_E, timeLU, label="LU decomposition")
plt.legend(loc="upper left")
plt.title("Plot of the dependence of algorithms times on the number N ")
plt.xlabel("Number N")
plt.ylabel("Time [s]")
plt.savefig('methods_plot.png')
plt.show()
