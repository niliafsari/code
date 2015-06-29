#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *

def matrify(a):
    L = np.swapaxes(a,1,2) #Swap axes such that reshape resizes correctly
    L = np.reshape(L,(L.shape[0]*L.shape[1],L.shape[2]*L.shape[3]))
    return np.matrix(L)

# A test block Toeplitz matrix, where a is the first column.
a = np.zeros(shape=(2,2),dtype=complex)
a_orig = np.zeros(shape=(2,2,2),dtype=complex)
c = np.zeros(shape=(2,2,2),dtype=complex)
a_fill = (1.4+0j, 0.432+0.132j, (0.229-0.318j)/4., (-0.137+0.187j)/2.)
A_orig = toeplitz(a_fill)

a_zpad = np.zeros(40,dtype=complex)
a_zpad[0:4] = a_fill
#a_orig[:,:,0] = toeplitz([1.4+0j, 0.432+0.132j])
#a_orig[:,:,1] = toeplitz([2, (-0.137+0.187j)/2.]) 
#a_orig[:,:,2] = toeplitz([25, 2j]) 
#print "a_orig=", a_orig[:,:,0], a_orig[:,:,1]

a[:,0] = [1.4+0j, 0.432+0.132j]
a[:,1] = [2, (-0.137+0.187j)/2.] 
#a[:,2]=[25, 2j]

#l1 = block_toeplitz_decomp(a)
l2 = toeplitz_decomp(np.array(a_fill))
#print "l1", l1
#print l1[:,:,2]
#print l2, l3

print l2,l2.shape
A = np.dot(l2,np.conj(l2).T)


#At = np.dot(l1[:,:,0],np.conj(l1[:,:,0]).T)
#Ap = np.dot(l1[:,:,1],np.conj(l1[:,:,1]).T)
#print "At=",aj, aj.shape, np.sum(a_orig,2).shape
#print np.dot(l3,np.conj(l3).T)
print("Consistency check, these numbers should be small:",np.sum(A-A_orig))
#print("Consistency check, these numbers should be small:",np.sum(aj-np.sum(a_orig)))

#print("Consistency check for block decomp., these numbers should be small:",np.sum(a_orig[:,:,0]-At)-np.sum(a_orig[:,:,1]-Ap))
