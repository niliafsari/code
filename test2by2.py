import sys
import numpy as np
from scipy.linalg import inv, toeplitz, hankel
from numpy import linalg as LA
#from mpi4py import MPI
#import h5py 
import time
from sp import multirate
from toeplitz_decomp import *
import matplotlib.pyplot as plt
from scipy import signal
if len(sys.argv) < 6:
    print "Usage: %s filename num_rows num_columns n b" % (sys.argv[0])
    sys.exit(1)

np.set_printoptions(precision=2, suppress=True, linewidth=200)
a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(int(sys.argv[2]),int(sys.argv[3])),order='F')
#a=np.load('specdata.npy')
#plt.subplot(1, 1, 1)
#plt.imshow(a.T, interpolation='nearest')
#plt.colorbar()
#plt.show()
n=int(sys.argv[4])
b=int(sys.argv[5])
pad=1
a_input=np.zeros(shape=(n+n*pad, b+b*pad), dtype=complex)

a_input[:n,:b]=np.copy(a[1501:1501+n,301:301+b])
a_input=np.where(a_input > 0, a_input, 0)
print a_input
del a

neff=n+n*pad
beff=b+b*pad
pad2=1
const=3

a_input=np.sqrt(a_input)
print a_input,"after sqrt"
#t= np.fft.fft2(a_input,s=(neff+4,beff+4))
#print t
print np.fft.fft2(a_input),"after first fftI"
a_input[:n,:b]=np.fft.fft2(a_input,s=(n,b))
#print a_input,"after first fft"

#a_input[0,beff/2+1:]=np.conj(a_input[0,1:beff/2][::-1])
#a_input[neff/2+1:,0]=np.conj(a_input[1:neff/2,0][::-1])
a_input[3*n/2:2*n,3*b/2:2*b]=a_input[n/2:n,b/2:b]
a_input[3*n/2:2*n,b/2:b]=a_input[n/2:n,b/2:b]
a_input[n/2:n,3*b/2:2*b]=a_input[n/2:n,b/2:b]
a_input[0:n/2,3*b/2:2*b]=a_input[0:n/2,b/2:b]
a_input[3*n/2:2*n,0:b/2]=a_input[n/2:n,0:b/2]
#a_input[1:neff/2,beff/2+1:beff]=np.fliplr(a_input[1:neff/2,1:beff/2])
#a_input[neff/2+1:neff,1:beff/2]=np.conj(np.flipud(a_input[1:neff/2,1:beff/2]))
#a_input[neff/2+1:neff,beff/2+1:beff]=np.conj(np.fliplr(np.flipud(a_input[1:neff/2,1:beff/2])))

#a_input[:,b/2:3*b/2]=np.zeros(shape=(neff,b))
#a_input[n/2:3*n/2,:]=np.zeros(shape=(n,beff))

print a_input,"after shift"
y=np.append(a_input[:,0:beff/2],np.zeros(shape=(beff,2)),axis=1)
y=np.append(y,a_input[:,beff/2:beff],axis=1)
z=np.append(y[0:neff/2,:],np.zeros(shape=(2,neff+2)),axis=0)
z=np.append(z,y[neff/2:,:],axis=0)
a_input=z
neff=neff+2
beff=beff+2
#x=np.sum(np.power(a_input[:,:],2))
#print x,"auto corr"
input_f=np.zeros(shape=(neff*(beff+pad2*beff), neff*(beff+pad2*beff)), dtype=complex)
corr=np.zeros(shape=(neff, beff), dtype=complex)
for i in xrange(0,neff):
	for j in xrange(0,beff):		
		temp = np.roll(a_input,j,axis=1)
		corr[i,j]=signal.correlate(a_input, np.roll(temp,i,axis=0),mode='valid')[0,0] /(neff*beff+0j)
		
print corr,"corr"

a_input=np.fft.ifft2(a_input,s=(neff,beff))
#a_input[:,beff/2]=np.zeros(neff)
#a_input[neff/2,:]=np.zeros(beff)
print a_input,"after second fft"
a_input=np.power(np.abs(a_input),2)
print a_input,"after abs^2"
a_input=np.fft.fft2(a_input,s=(neff,beff)) 
#print corr/a_input,"factor"
print a_input,"after third fft"
#print np.fft.ifft2(a_input),"ifft"
#temp=a_input[1:neff, beff/2:beff]
#temp=np.flipud(temp)
#a_input[1:neff, beff/2:beff]=temp
#print a_input,"after third fft"
#x=a_input[neff/2-1,:]
#a_input[neff-1,:]=np.conj(x)
#print a_input,"after third fft"
for i in xrange(0,neff):
	for j in xrange(i,neff):
		#input_f[i*(beff+pad2*beff):(i+1)*(beff+pad2*beff),j*(beff+pad2*beff):(j+1)*(beff+pad2*beff)]=toeplitz(np.append(a_input[j-i,:],np.zeros(pad2*beff)))
		if np.abs(i-j)<=neff/2:
			rows=np.append(a_input[j-i,:beff-const],np.zeros(pad2*beff+const))
			cols=np.append(np.append(a_input[j-i,0],a_input[j-i,const+1:][::-1]),np.zeros(pad2*beff+const))
			input_f[i*(beff+pad2*beff):(i+1)*(beff+pad2*beff),j*(beff+pad2*beff):(j+1)*(beff+pad2*beff)]=toeplitz(cols,rows)
			#print i,j,hankel(rows,cols)
		elif np.abs(j-i)>neff/2 and const:
			input_f[i*(beff+pad2*beff):(i+1)*(beff+pad2*beff),j*(beff+pad2*beff):(j+1)*(beff+pad2*beff)]=np.zeros(shape=(beff+pad2*beff,beff+pad2*beff))
		else:
			input_f[i*(beff+pad2*beff):(i+1)*(beff+pad2*beff),j*(beff+pad2*beff):(j+1)*(beff+pad2*beff)]=toeplitz(np.append(a_input[j-i,:beff-const],np.zeros(pad2*beff+const)))	
			#input_f[i*(beff+pad2*beff):(i+1)*(beff+pad2*beff),j*(beff+pad2*beff):(j+1)*(beff+pad2*beff)]=np.conj(input_f[(0)*(beff+pad2*beff):(1)*(beff+pad2*beff),(beff/2-1)*(beff+pad2*beff):(beff/2)*(beff+pad2*beff)].T)
input_f=np.conj(np.triu(input_f).T)+np.triu(input_f,1)
print input_f[0:12,0:12], "blocks00"
print input_f[0:12,12:24],"blocks01"
print input_f[0:12,24:36], "block02"
print input_f[0:12,36:48],"block03"
print input_f[0:12,48:60],"block04"
print input_f[0:12,60:72],"block05"
#print input_f[0:8,16:32], neff,(beff+pad2*beff)
print np.sum(input_f, axis=1),"sums"
epsilon=np.identity(neff*(beff+pad2*beff))*np.max(np.abs(np.linalg.eigvals(input_f)))*10e-10
input_f=epsilon+input_f
w, v = LA.eig(input_f)
print w,"values"
print v,"vectors"
#print np.linalg.eigvals(input_f)
#input_f=input_f+epsilon
L=np.linalg.cholesky(input_f)
#output_file="real_n_%s_b_%s_neff_%s_beff_%s.dat" %(str(n),str(b),str(neff),str(beff+pad2*beff))
#output = np.memmap(output_file, dtype='complex', mode='w+', shape=(beff+pad2*beff, neff*(beff+pad2*beff)),order='F')
#output[:,:]=input_f[:beff+pad2*beff,:]
#del output

l=toeplitz_blockschur(input_f,(beff+pad2*beff),0)
print l
