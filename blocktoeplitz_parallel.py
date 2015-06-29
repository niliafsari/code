#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from scipy.linalg import toeplitz
from toeplitz_decomp import *

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

#Create block sizes for MPI processes
                                                                                                                                                                                                                                                     
g=np.zeros(shape=(2*b,n*b), dtype=complex)
if rank==0:
	l=np.zeros(shape=(n*b,n*b), dtype=complex)
c=myBlockChol(a[0:b,0:b],1)
for j in xrange(0,n): 
     g[b:2*b,j*b:(j+1)*b]= -np.dot(inv(c),a[0:b,j*b:(j+1)*b])      
     g[0:b,j*b:(j+1)*b]= -g[b:2*b,j*b:(j+1)*b]
     l[0:b,j*b:(j+1)*b] = g[0:b,j*b:(j+1)*b] 
g[0:b,b:n*b]=g[0:b,0:(n-1)*b]  
g[0:b,0:b]=np.zeros(shape=(b,b), dtype=complex)
g[b:2*b,0:b]=np.zeros(shape=(b,b), dtype=complex)
for i in xrange( 1, n):
  size_per_node = (n-i)*b//size
  if size_per_node>=b:
	size_per_node = (n-i)*b//size 
	size_this_node=size_per_node
	if rank==size-1:
		size_this_node = (n-i)*b//size + (n-i)*b%size
	start = i*b+rank*size_per_node
	end = min(start+size_this_node, n*b)
  else:
	size_per_node = (n-i-1)*b//(size-1) 
	size_this_node=size_per_node
  	if rank==size-1:
		size_this_node = (n-i-1)*b//(size-1) + (n-i-1)*b%(size-1)
	if (rank==0):
		start = i*b
		end = min(start+b, n*b)
	else:
		start = (i+1)*b+(rank-1)*size_per_node
		end = min(start+size_this_node, n*b)
  for j in xrange(0,b):
	if rank==0:
		g02=np.copy(g[b:2*b,start+j])	
		g01=g[j,start+j]
	else:
		g02=None
		g01=None
	comm.bcast([g02 MPI.Complex],root=0)
	comm.bcast([g01 MPI.Complex],root=0)
	if g02==0:
		g[b:2*b,start:end]=-g[b:2*b,start:end]
		continue				
	sigma=np.dot(np.conj(g02.T),g02)
	alpha=-np.sign(g01)*np.sqrt(g01**2 - sigma)
	z=g01+alpha
	x2=-np.copy(g02)/np.conj(z)
	beta=(2*z*np.conj(z))/(np.conj(z)*z-sigma)
	if rank==0 :
		g[j,start+j]=-alpha
		g[b:2*b,start+j]=0
		v=np.copy(g[j,start+j+1:end]+np.dot(np.conj(x2.T),g[b:2*b,start+j+1:end]))
		g[j,start+j+1:end]=g[j,start+j+1:end]-beta*v
		v=np.reshape(v,(1,v.shape[0]))
		x2=np.reshape(x2,(1,x2.shape[0]))
		g[b:2*b,start+j+1:end]=-g[b:2*b,start+j+1:b*n]-beta*np.dot(x2.T,v)
	else:
		v=np.copy(g[j,start:end]+np.dot(np.conj(x2.T),g[b:2*b,start:end]))
		g[j,start:end]=g[j,start:end]-beta*v
		v=np.reshape(v,(1,v.shape[0]))
		x2=np.reshape(x2,(1,x2.shape[0]))
		g[b:2*b,start:end]=-g[b:2*b,start:end]-beta*np.dot(x2.T,v)
  G=np.zeros_like(g)
  comm.Allreduce(g, G, op=MPI.SUM)
  if (rank==0):
  	l[i*b: (i+1)*b,0:n*b]=G[0:b,0:n*b]
  g[0:b,i*b:n*b]=G[0:b,(i-1)*b:(n-1)*b]
  g[0:b,(i-1)*b:i*b]=np.zeros(shape=(b,b), dtype=complex)

