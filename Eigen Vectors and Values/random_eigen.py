#Fahmid - 1705087
import math
import random
import numpy as np

print("\nFahmid = 1705087")

n = int(input("\nEnter the value of n: "))

A = np.random.randint(0, 100, size=(n,n))

while np.linalg.det(A) == 0:
    A = np.random.randint(0, 100, size=(n,n))
    A = (A + A.T)

print("\nA = \n", A)

eigen_values, eigen_vectors = np.linalg.eig(A)

print("\nEigen values = \n", eigen_values)
print("\nEigen vectors = \n", eigen_vectors)

A_reconstructed = np.dot(eigen_vectors, np.dot(np.diag(eigen_values), np.linalg.inv(eigen_vectors)))

print("\nA_reconstructed = \n", A_reconstructed)

print("\nIs the reconstruction perfect? ", np.allclose(A, A_reconstructed))

print("\n")


