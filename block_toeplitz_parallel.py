#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from scipy.linalg import inv, toeplitz

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

# A test block Toeplitz matrix, where a is the first column.
a = np.zeros(shape=(5000,2,2),dtype=complex)
a[0] = [ [1.4+0j , 0+0j] , [0+0j , 1.4+0j] ]
a[1] = [ [0.432+0.132j , 0+0j] , [0+0j , 0.432+0.132j] ]

n=a.shape[0]	
b=a.shape[1]	
l = np.zeros(shape=(n,n,b,b), dtype=complex)
e = np.zeros(shape=(1,b,b), dtype=complex)

#Create block sizes for MPI processes
size_per_node = (n-1)//size + 1
start = rank*size_per_node
end = min((rank+1)*size_per_node, n)

alpha=np.concatenate( (-np.conj(a[1:,:,:]),e), axis=0 )[start:end,:,:]
beta=np.conj(a)[start:end,:,:]

for i in xrange(n):

    if rank == 0:
        #Calculate gamma and s once and broadcast
        gamma = np.dot( alpha[0] , inv(np.real(beta[0])) )
        s = np.sqrt(np.real(beta[0]))

    else:
        gamma = None
        s = None

    gamma = comm.bcast(gamma,root=0)
    s = comm.bcast(s,root=0)

    #if np.real(beta[0]) < 0:
    #    print("ERROR - not positive definite")
    #l[i:n,i] = np.dot( np.conj(beta)[0:n-i] , inv(s) )

    beta -= np.swapaxes( np.dot(np.conj(gamma), alpha), 0, 1)
    alpha -= np.swapaxes( np.dot(gamma, beta), 0, 1 )

    if rank != 0:
        alpha_endpt = np.reshape(alpha[0,:,:],(1,b,b))
        comm.send(alpha_endpt,dest=(rank-1))

    if rank != (size-1):        
        alpha_endpt = comm.recv(source=(rank+1))
        alpha = np.concatenate( (alpha[1:,:,:],alpha_endpt), axis=0)
    else:
        alpha = np.concatenate( (alpha[1:,:,:],e), axis=0 )

if rank == 0:
    print("beta[0][0]: ", np.dot(beta[0],inv(s)))

#L = np.swapaxes(a,1,2) #Swap axes such that reshape resizes correctly
#L = np.reshape(L,(L.shape[0]*L.shape[1],L.shape[2]*L.shape[3]))
#L = np.matrix(L)

#A = L*np.conj(L.T)
#print A
#print("Consistency check, these numbers should be small:",np.sum(A-A_orig))
