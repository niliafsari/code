#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from new_parallel import *
from mpi4py import MPI

# 1-d case
a=np.load('simdata.npy')
A=np.copy(toeplitz(a[:,1]))
A_orig=np.copy(toeplitz(a[:,1]))
l=block_toeplitz_par(A,1)
result = np.dot(np.conj(l).T,l)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
	result = np.dot(np.conj(l).T,l)
	print("Consistency check, these numbers should be small:",np.sum(A_orig-result[0:512,0:512]))
	

if rank==0:
	l=toeplitz_blockschur(np.conj(a[:,1:2].T),1,0)
	result = np.dot(np.conj(l).T,l)
	print("Consistency check, these numbers should be small:",np.sum(A_orig-result[0:512,0:512]))