
import numpy as np

# just computes the matrix to show the configuration-boundary
# of 1024 hash functions

B = np.zeros((51,51))

for i in range(1,50):
    for j in range(1,50):
        if i*j<=1024:
            B[i,j] = 1.0

np.save("B.npy", B)