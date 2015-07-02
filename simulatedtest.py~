#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *

# 1-d case
a=np.load('simdata.npy')
A=toeplitz(a[:,0])
in0=np.zeros(shape=(512,1), dtype=complex)
in0=np.copy(a[0:512,0:1])
#l=toeplitz_blockschur(np.conj(a[:,0:1].T),1)
l = toep_zpad(np.array(a[0:512,0]).reshape(-1,).tolist(),2)
result = np.dot(l,np.conj(l).T)
print("Consistency check, these numbers should be small:",np.sum(A-result[0:512,0:512]))

# 1-d case
l=toeplitz_blockschur(np.conj(a[:,0:1].T),1,2)
result = np.dot(np.conj(l).T,l)
print("Consistency check, these numbers should be small:",np.sum(A-result[0:512,0:512]))