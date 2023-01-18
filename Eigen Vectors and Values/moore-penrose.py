#Fahmid - 1705087
import math
import random
import numpy as np

print("\nFahmid = 1705087")

n = int(input("\nEnter the value of n: "))
m = int(input("\nEnter the value of m: "))

A = np.random.randint(0, 100, size=(n,n))

print("\nA = \n", A)

U, D, VT = np.linalg.svd(A)

print("\nU = \n", U)
print("\nD = \n", D)
print("\nVT = \n", VT)

D_inver = np.linalg.inv(np.diag(D))
print("\nD_inver = \n", D_inver)

if n>m:
    Dplus = np.concatenate((D_inver, np.zeros((m,n-m))),axis = 1)
else:
    Dplus = np.concatenate((D_inver, np.zeros((m-n,n))),axis = 0)

print("\nDplus = \n", Dplus)

Aplus = np.dot(VT.T, np.dot(Dplus,U.T))
print("\nAplus = \n", Aplus)

pseudoinverse = np.linalg.pinv(A)
print("\npseudoinverse numpy= \n", pseudoinverse)

print("\nIs the inverses are equal? ", np.allclose(pseudoinverse, Aplus))





