#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp_original import *

def matrify(a):
    L = np.swapaxes(a,1,2) #Swap axes such that reshape resizes correctly
    L = np.reshape(L,(L.shape[0]*L.shape[1],L.shape[2]*L.shape[3]))
    return np.matrix(L)

# A test block Toeplitz matrix, where a is the first column.
a = np.zeros(shape=(2,2,2),dtype=complex)
a_fill = (1.4+0j, 0.432+0.132j, (0.229-0.318j)/4., (-0.137+0.187j)/2.)
A_orig = toeplitz(a_fill)
print A_orig
a_zpad = np.zeros(40,dtype=complex)
a_zpad[0:4] = a_fill
a[0] = toeplitz([10, 1])
a[1] = toeplitz( [5, 1-1.0j] )

l1 = block_toeplitz_decomp(a)
l2 = toeplitz_decomp(np.array(a_fill))
print l1[0,0,:,:]
print l1[0,1,:,:]
print l1[1,0,:,:]
print l1[1,1,:,:]
#print np.dot(l1[:,:,0,0],l1[:,:,0,0].T), l1.shape
#print l2, l3
L=matrify(l1)
print L,L.shape
A = np.dot(l2,np.conj(l2).T)
#print np.dot(L,np.conj(L).T)
print("Consistency check, these numbers should be small:",np.sum(A-A_orig))
