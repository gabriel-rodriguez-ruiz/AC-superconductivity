#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:31:26 2024

@author: gabriel
"""

from scipy.linalg import solveh_banded
import numpy as np
import scipy
import random
import time

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def sp_inv(A, b):
    N = np.shape(A)[0]
    D = int((np.count_nonzero(A[0,:])-1))
    ab = np.zeros((2*D+1,N))
    for i in np.arange(0, D):
        ab[i,:] = np.concatenate((np.zeros(D-i,), np.diag(A,k=i-D)),axis=None)
        ab[2*D-i,:] = np.concatenate((np.diag(A,k=2*D-1-i), np.zeros(D-i,)),axis=None)
    ab[D,:] = np.diag(A,k=0)
    y = solveh_banded((D,D), ab,b)
    return y

N = 4
array = np.random.rand(N-1)
A = np.diag(array, k=-1) + np.diag(np.random.rand(N), k=0) + np.diag(array, k=1)
I = np.eye(N)

A_inverted = np.zeros_like(A)

start = time.time()

for i in range(N):
    A_inverted[:, i] = sp_inv(A, I[:, i])

print(time.time()-start)

start = time.time()

A_inverted_scipy = scipy.linalg.inv(A)

print(time.time()-start)
