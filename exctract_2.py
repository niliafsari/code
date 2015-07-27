#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from mpi4py import MPI

# 1-d case
a=np.load('specdata.npy')
n=8
b=4
pad=1
a_input=np.zeros(shape=(n+n*pad, b+b*pad), dtype=complex)
neff=n+n*pad
beff=b+b*pad
a_input[:n,:b]=a[:n,:b]
a_input=np.sqrt(a_input)
a_input[:n,:b]=np.fft.fft2(a_input,s=(n,b))
a_input[3*n/2:2*n,3*b/2:2*b]=a_input[n/2:n,b/2:b]
a_input[0:n/2,3*b/2:2*b]=a_input[0:n/2,b/2:b]
a_input[3*n/2:2*n,0:b/2]=a_input[n/2:n,0:b/2]
a_input[:,b/2:3*b/2]=np.zeros(shape=(neff,b))
a_input[n/2:3*n/2,:]=np.zeros(shape=(n,beff))
a_input=np.fft.ifft2(a_input,s=(neff,beff))
a_input=np.power(np.abs(a_input),2)
a_input=np.fft.fft2(a_input,s=(neff,beff))
print a_input.shape
#for i in xrange(0,n):
	#print i
	#l = toeplitz_decomp(np.array(cfact[i,0:ns]).reshape(-1,).tolist())
	#l = toeplitz_decomp(np.array(cfact[:,i]).reshape(-1,).tolist())
np.save('simple_fft2_simdata.npy',a_input)



#l=toeplitz_blockschur(a_input,b,pad)
#if rank==0:
#result = np.dot(np.conj(l).T,l)
#print("Consistency check, these numbers should be small:",np.sum(a_input-result[0:b,:]))
