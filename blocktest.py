#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from new_parallel import *
from mpi4py import MPI

# 1-d case
a=np.load('simple_fft_simdata.npy')
print a.shape
n=a.shape[0]
b=a.shape[1]
pad=0
a_input=np.zeros(shape=(b,b*n), dtype=complex)
for i in xrange(0,n):
	inner_block=toeplitz(a[i,:])
	a_input[:,i*b:(i+1)*b]=np.conj(inner_block.T)
aa_input=np.zeros(shape=(b*n,b*n), dtype=complex)
pad2=0
for i in xrange(0,n):
	for j in xrange(i,n):
		aa_input[(j-i)*(b+pad2*b):(j-i)*(b+pad2*b)+b,j*(b+pad2*b):j*(b+pad2*b)+b]=toeplitz(a[i,:])
		

aa_input=np.conj(np.triu(aa_input,1).T)+aa_input
input_ff=np.zeros(shape=(2*b*n,2*b*n), dtype=complex)
input_ff[:n*(b+pad2*b),:n*(b+pad2*b)]=aa_input
L = np.linalg.cholesky(aa_input)


#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#print a_input.shape
#l=toeplitz_blockschur(a_input,b,pad)
#if rank==0:
#result = np.dot(np.conj(l).T,l)
#print("Consistency check, these numbers should be small:",np.sum(a_input-result[0:b,:]))

