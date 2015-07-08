
import numpy as np
from mpi4py import MPI
from scipy.linalg import toeplitz
from toeplitz_decomp import *
import itertools
def block_toeplitz_par(a,b,pad):
	comm = MPI.COMM_WORLD
	size  = comm.Get_size()
	rank = comm.Get_rank()
	n=a.shape[1]/b
	size_node_temp=(n//size)*b
	size_node=size_node_temp
	if rank==size-1:
		size_node = (n//size)*b + (n%size)*b
	start = rank*size_node_temp
	end = min(start+size_node, n*b)
	g1=np.zeros(shape=(b,size_node), dtype=complex)
	g2=np.zeros(shape=(b,size_node), dtype=complex)
	temp=np.zeros(shape=(b,b), dtype=complex)
	if rank==0:
		l=np.zeros(shape=((n+n*pad)*b,(n+n*pad)*b), dtype=complex)
	else:
		l=np.zeros(shape=(1,1), dtype=complex)
	#for simulated data that the first matrix is toeplitz
	#c=toeplitz_decomp(np.array(a[0:b,0]).reshape(-1,).tolist())
	c=myBlockChol(a[0:b,0:b],1)
	for j in xrange(0,size_node/b): 
		g2[:,j*b:(j+1)*b]= -np.dot(inv(c),a[0:b,start+j*b:start+(j+1)*b]) 
		g1[:,j*b:(j+1)*b]= -g2[:,j*b:(j+1)*b]
	empty=0
	data = comm.gather(g1[:,:], root=0)
	if rank==0:
		l[0:b,0:n*b]=np.concatenate(data, axis=1)
	global_start_g1=0
	for i in xrange( 1,n + n*pad):    
		global_end_g1=min(n*b,(n+n*pad-i)*b)
		start_g1=start
		if (global_end_g1<end and start<global_end_g1):
			end_g1=global_end_g1	
		elif (global_end_g1<end and start>=global_end_g1):
			empty=1
			g1=np.zeros_like(g1)
			end_g1=start
		else: 
			end_g1=end
		length_g=end_g1-start_g1
		if  rank !=size-1:
			comm.Recv(temp,source=rank+1,tag=i*size+rank)
		if  rank !=0: 
			data=np.copy(g2[:,0:b])
			comm.Send(data,dest=rank-1,tag=i*size+rank-1)
		g2[:,0:size_node-b]=g2[:,b:size_node]
		if  rank !=size-1:
			g2[:,size_node-b:size_node]=temp
		if (i<n*pad+1) and (rank==size-1):
			g2[:,size_node-b:size_node]=np.zeros(shape=(b,b))
		for j in xrange(0,b):
			if rank==0:
				g0_1=np.copy(g1[j,j])
				g0_2=np.copy(g2[:,j])
			else:
				g0_1=np.zeros(shape=(1,1), dtype=complex)
				g0_2=np.zeros(shape=(b,1), dtype=complex)
			g0_1=comm.bcast(g0_1 ,root=0)		
			g0_2=comm.bcast(g0_2,root=0)
			if empty:
				continue
			if g0_2.all()==0:
				g2[:,0:length_g]=-g2[:,0:length_g]
				continue		
			sigma=np.dot(np.conj(g0_2.T),g0_2)
			alpha=-np.sign(g0_1)*np.sqrt(g0_1**2.0 - sigma)
			z=g0_1+alpha
			x2=-np.copy(g0_2)/np.conj(z)
			beta=(2*z*np.conj(z))/(np.conj(z)*z-sigma)
			if rank==0 :
				g1[j,j]=-alpha
				g2[:,j]=0
				v=np.copy(g1[j,j+1:length_g]+np.dot(np.conj(x2.T),g2[:,j+1:length_g]))
				g1[j,j+1:length_g]=g1[j,j+1:length_g]-beta*v
				v=np.reshape(v,(1,v.shape[0]))
				x2=np.reshape(x2,(1,x2.shape[0]))
				g2[:,j+1:length_g]=-g2[:,j+1:length_g]-beta*np.dot(x2.T,v)
			else:
				v=np.copy(g1[j,:]+np.dot(np.conj(x2.T),g2))
				g1[j,:]=g1[j,:]-beta*v
				v=np.reshape(v,(1,v.shape[0]))
				x2=np.reshape(x2,(1,x2.shape[0]))
				g2=-g2-beta*np.dot(x2.T,v)
		data = comm.gather(g1[:,0:length_g], root=0)
		if rank==0:
			l[i*b:(i+1)*b,i*b:i*b+global_end_g1]=np.concatenate(data, axis=1)
	return l
