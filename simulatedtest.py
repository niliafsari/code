#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *

# 1-d case
a=np.load('simdata.npy')
pad=1
b=1
n=a.shape[0]
print n
A=toeplitz(a[:,0])
z=np.zeros(shape=((n+pad*n)*b,(n+pad*n)*b), dtype=complex)
z[0:n*b,0:n*b]=toeplitz(a[:,0])
#A=toeplitz(z[:,0])
#in0=np.copy(a[0:512,0:1])
#toeplitz_decomp
#l=toeplitz_blockschur(np.conj(a[:,0:1].T),1)
l = toep_zpad(np.array(z[:,0:1]).reshape(-1,).tolist(),0)
k=np.copy(l[n+n*pad-1,n*pad:n+n*pad])
u=np.copy(l[n+n*pad-1,0:n+n*pad])
ll=np.correlate(u,k)[1:n+1]
ll=ll[::-1]
print sum(ll[:]-a[:,0])
#result = np.dot(l,np.conj(l).T)
AA=toeplitz(np.conj(ll[0:n]))
print("Consistency check, these numbers should be small:",np.sum(AA-A))

# 1-d case
#l=toeplitz_blockschur(np.conj(a[:,0:1].T),1,0)
#result = np.dot(np.conj(l).T,l)
#print("Consistency check, these numbers should be small:",np.sum(A-result[0:512,0:512]))