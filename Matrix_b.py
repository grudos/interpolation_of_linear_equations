import numpy as np

from globals import *
from math import sin

class Matrix_b:
    def __init__(self, N):
        self.N = N
        self.b = np.zeros((N, 1))

        for i in range(self.N):
            self.b[i][0] = sin(i * (f + 1))

    def getMatrix_b(self):
        return self.b