import numpy as np

from globals import *

class Matrix_A:
    def __init__(self, a1, N):
        self.N = N
        self.A = np.zeros((N, N))
        self.a1 = a1

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.A[i][j] = self.a1
                elif j == i - 1 or j == i + 1:
                    self.A[i][j] = a2
                elif j == i - 2 or j == i + 2:
                    self.A[i][j] = a3

    def getMatrix_A(self):
        return self.A



