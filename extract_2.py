import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from mpi4py import MPI
#import h5py 
import time
from sp import multirate
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# 1-d case
a=np.load('specdata.npy')
n=2
b=2
pad=1
a_input=np.zeros(shape=(n+n*pad, b+b*pad), dtype=complex)
a_input[:n,:b]=a[:n,:b]
#a_input[:n,:b]=1
neff=n+n*pad
beff=b+b*pad
pad2=2
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
#a_input[np.where(np.abs(a_input)<10e-7)]=2e-2
#print a_input.reshape(1,a_input[0]*a_input[1])
for i in xrange(0,neff):
	for j in xrange(i,neff):
		input_f[(j-i)*(beff+pad2*beff):(j-i+1)*(beff+pad2*beff),j*(beff+pad2*beff):(j+1)*(beff+pad2*beff)]=toeplitz(np.append(a_input[i,:],np.zeros(pad2*beff)))
input_f=np.conj(np.triu(input_f).T)+np.triu(input_f,1)		
#epsilon=np.identity(neff*(beff+pad2*beff))*np.max(np.linalg.eigvals(input_f))*10e-10
#input_f=input_f+epsilon
#print  input_f.shape
#input_ff=np.zeros(shape=(2*neff*(beff+pad2*beff), 2*neff*(beff+pad2*beff)), dtype=complex)
#input_ff[:neff*(beff+pad2*beff),:neff*(beff+pad2*beff)]=input_f
#epsilon=np.identity(2*neff*(beff+pad2*beff))*np.max(np.linalg.eigvals(input_f))*10e-10
#input_ff=input_ff+epsilon
#t = time.time()
l=toeplitz_blockschur(input_f,(beff+pad2*beff),1)
#ll=toeplitz_blockschur(input_ff,(beff+pad2*beff),0)
#print("Consistency check, these numbers should be small:",np.sum(ll-l))
#elapsed = time.time() - t
#print elapsed
#L=np.linalg.cholesky(input_ff)
#print L.shape
#print np.conj(L[l.shape[0]-62,0:2*neff*(beff+pad2*beff)])
#result =np.dot(L,L.T.conj())   
print input_f[:,0]        
#ll[:,ll.shape[1]-6:ll.shape[1]-1]      
#result =np.dot(L,L.T.conj())
result = np.dot(np.conj(l).T,l)
k=np.copy(l[neff*(beff+pad2*beff):2*neff*(beff+pad2*beff),l.shape[0]-1])
u=np.copy(l[0:2*neff*(beff+pad2*beff),l.shape[0]-1])
#u=np.copy(l[n+n*pad-1,0:n+n*pad])


lll=np.correlate(u,np.conj(k))
lll=lll[::-1]
q=np.zeros(0)
for i in xrange(0,beff*3,3):
	q=np.append(q,range(i*beff+beff,i*beff+beff*3),0)
lll=np.delete(lll, q)
print lll
#k=np.copy(l[neff*(beff+pad2*beff):2*neff*(beff+pad2*beff),l.shape[0]-5])
#u=np.copy(l[0:2*neff*(beff+pad2*beff),l.shape[0]-5])
#print u
#u=np.copy(l[n+n*pad-1,0:n+n*pad])
#lll=np.correlate(u,np.conj(k))
#lll=lll[::-1]
#print(lll) 
#print("Consistency check, these numbers should be small:",np.sum(result-input_ff))
#np.save('simple_fft2_simdata.npy',input_ff)
#if rank==0:
#result = np.dot(np.conj(l).T,l)
#print("Consistency check, these numbers should be small:",np.sum(input_f-result[0:beff,0:neff*beff]))


#print l[:,(neff+3*neff)*beff-beff:(neff+3*neff)*beff]

#l = toep_zpad(np.array(z[:,0:1]).reshape(-1,).tolist(),0)
#k=np.copy(l[n+n*pad-1,n*pad:n+n*pad])
#u=np.copy(l[n+n*pad-1,0:n+n*pad])
#ll=np.correlate(u,k)[1:n+1]
#ll=ll[::-1]
