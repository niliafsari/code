#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from new_parallel import *
from mpi4py import MPI

# 1-d case
a=np.load('simdata.npy')
#a_orig=np.copy(toeplitz(a[:,:]))
aa=np.copy(a[:,0:4])
#l = toeplitz_decomp(np.array(aa).reshape(-1,).tolist())
ns=aa.shape[1]
n=aa.shape[0]
#for i in xrange(0,n):
#	print i
#	l = toeplitz_decomp(np.array(a[:,i]).reshape(-1,).tolist())
cfact=np.zeros(shape=(n,2*ns), dtype=complex)
cfact[:,0:ns]=np.sqrt(aa)
cfact[:,0:ns]=np.fft.fft(cfact,ns)
cfact[:,3*ns/2:2*ns]=cfact[:,ns/2:ns]
cfact[:,ns/2:3*ns/2]=np.zeros(shape=(1,ns))
cfact=np.fft.ifft(cfact,2*ns)
cfact=np.power(np.abs(cfact),2)
cfact=np.fft.fft(cfact,2*ns)
print cfact.shape
#for i in xrange(0,n):
	#print i
	#l = toeplitz_decomp(np.array(cfact[i,0:ns]).reshape(-1,).tolist())
	#l = toeplitz_decomp(np.array(cfact[:,i]).reshape(-1,).tolist())
np.save('simple_fft_simdata.npy',cfact)


