#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from mpi4py import MPI
import h5py 

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# 1-d case
a=np.load('specdata.npy')
n=32
b=8
pad=1
a_input=np.zeros(shape=(n+n*pad, b+b*pad), dtype=complex)
a_input[:n,:b]=a[:n,:b]
neff=n+n*pad
beff=b+b*pad
pad2=0
input_f=np.zeros(shape=(neff*(beff+pad2*beff), neff*(beff+pad2*beff)), dtype=complex)
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
#print a_input
for i in xrange(0,neff):
	for j in xrange(i,neff):
		input_f[(j-i)*(beff+pad2*beff):(j-i)*(beff+pad2*beff)+beff,j*(beff+pad2*beff):j*(beff+pad2*beff)+beff]=toeplitz(a_input[i,:])
		

input_f=np.conj(np.triu(input_f,1).T)+input_f
input_ff=np.zeros(shape=(2*neff*(beff+pad2*beff), 2*neff*(beff+pad2*beff)), dtype=complex)
input_ff[:neff*(beff+pad2*beff),:neff*(beff+pad2*beff)]=input_f
L = np.linalg.cholesky(input_ff)
np.save('simple_fft2_simdata.npy',input_f)


#l=toeplitz_blockschur(input_f,beff+pad2*beff,3)
#if rank==0:
#result = np.dot(np.conj(l).T,l)
#print("Consistency check, these numbers should be small:",np.sum(input_f-result[0:beff,0:neff*beff]))


#print l[:,(neff+3*neff)*beff-beff:(neff+3*neff)*beff]

#l = toep_zpad(np.array(z[:,0:1]).reshape(-1,).tolist(),0)
#k=np.copy(l[n+n*pad-1,n*pad:n+n*pad])
#u=np.copy(l[n+n*pad-1,0:n+n*pad])
#ll=np.correlate(u,k)[1:n+1]
#ll=ll[::-1]