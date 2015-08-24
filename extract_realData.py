#from numba import jit
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


#@jit
np.set_printoptions(precision=2, suppress=True, linewidth=200)
if len(sys.argv) < 8:
    print "Usage: %s filename num_rows num_columns offsetn offsetm sizen sizem" % (sys.argv[0])
    sys.exit(1)

num_rows=int(sys.argv[2])
num_columns=int(sys.argv[3])
offsetn=int(sys.argv[4])
offsetm=int(sys.argv[5])
sizen=int(sys.argv[6])
sizem=int(sys.argv[7])

if offsetn>num_rows or offsetm>num_columns or offsetn+sizen>num_rows or offsetm+sizem>num_columns:
	print "Error sizes or offsets don't match"
	sys.exit(1)

a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')

#plt.subplot(1, 1, 1)
#plt.imshow(a.T, interpolation='nearest')
#plt.colorbar()
#plt.show()

pad=1
pad2=1
debug=0

neff=sizen+sizen*pad
meff=sizem+sizem*pad

a_input=np.zeros(shape=(neff,meff), dtype=complex)
a_input[:sizen,:sizem]=np.copy(a[offsetn:offsetn+sizen,offsetm:offsetm+sizem])
del a

a_input=np.where(a_input > 0, a_input, 0)
const=pad2*neff/2
a_input=np.sqrt(a_input)
if debug:
	print a_input,"after sqrt"

a_input[:sizen,:sizem]=np.fft.fft2(a_input,s=(sizen,sizem))

if debug:
	print a_input,"after first fft"

a_input[sizen/2:sizen,:sizem]=np.zeros(shape=(sizen/2,sizem))
a_input[:sizen,sizem/2:sizem]=np.zeros(shape=(sizen,sizem/2))
a_input[neff-(sizen/2-1):neff,0]=a_input[1:sizen/2,0][::-1]
a_input[0,meff-(sizem/2-1):meff]=a_input[0,1:sizem/2][::-1]
a_input[neff-(sizen/2-1):neff,meff-(sizem/2-1):meff]=np.fliplr(np.flipud(a_input[1:sizen/2,1:sizem/2]))

if debug:
	print a_input,"after shift"


#corr=np.zeros(shape=(neff, meff), dtype=complex)
#for i in xrange(0,neff):
#	for j in xrange(0,meff):		
#		temp = np.roll(a_input,j,axis=1)
#		corr[i,j]=signal.correlate(a_input, np.roll(temp,i,axis=0),mode='valid')[0,0] /(neff*meff+0j)	
#print corr,"corr"

a_input=np.fft.ifft2(a_input,s=(neff,meff))
if debug:
	print a_input,"after second fft"
a_input=np.power(np.abs(a_input),2)
if debug:
	print a_input,"after abs^2"
a_input=np.fft.fft2(a_input,s=(neff,meff)) 
if debug:
	print a_input,"after third fft"

input_f=np.zeros(shape=(sizen*(meff+pad2*meff), sizen*(meff+pad2*meff)), dtype=complex)
for i in xrange(0,neff/2):
	for j in xrange(i,neff/2):
		if j>i:
			rows=np.append(a_input[j-i,:meff-const],np.zeros(pad2*meff+const))
			cols=np.append(np.append(a_input[j-i,0],a_input[j-i,const+1:][::-1]),np.zeros(pad2*meff+const))
			input_f[i*(meff+pad2*meff):(i+1)*(meff+pad2*meff),j*(meff+pad2*meff):(j+1)*(meff+pad2*meff)]=toeplitz(cols,rows)
		else:
			input_f[i*(meff+pad2*meff):(i+1)*(meff+pad2*meff),j*(meff+pad2*meff):(j+1)*(meff+pad2*meff)]=toeplitz(np.append(a_input[j-i,:meff-const],np.zeros(pad2*meff+const)))	

input_f=np.conj(np.triu(input_f).T)+np.triu(input_f,1)

if debug:
	print input_f[0:meff+pad2*meff,0:meff], "blocks00"
	print input_f[0:meff+pad2*meff,meff+pad2*meff:(meff+pad2*meff)+(meff)],"blocks01"
	print input_f[0:meff+pad2*meff,2*(meff+pad2*meff):2*(meff+pad2*meff)+(meff)],"blocks02"
	print input_f[0:meff+pad2*meff,3*(meff+pad2*meff):3*(meff+pad2*meff)+(meff)],"blocks03"
#print input_f[0:meff+pad2*meff,4*(meff+pad2*meff):4*(meff+pad2*meff)+(meff)],"blocks04"
#print input_f[0:meff+pad2*meff,5*(meff+pad2*meff):5*(meff+pad2*meff)+(meff)],"blocks05"
#print input_f[0:8,16:32], neff,(meff+pad2*meff)
#print np.sum(input_f, axis=1),"sums"
epsilon=np.identity(neff*(meff+pad2*meff)/2)*np.max(np.abs(np.linalg.eigvals(input_f)))*10e-10
input_f=epsilon+input_f
#w, v = LA.eig(input_f)
#print w,"values"
#print v,"vectors"
#print np.linalg.eigvals(input_f)
#input_f=input_f+epsilon
#L=np.linalg.cholesky(input_f)
#output_file="real_n_%s_b_%s_neff_%s_meff_%s.dat" %(str(n),str(b),str(neff),str(meff+pad2*meff))
#output = np.memmap(output_file, dtype='complex', mode='w+', shape=(meff+pad2*meff, neff*(meff+pad2*meff)),order='F')
#output[:,:]=input_f[:meff+pad2*meff,:]
#del output
if debug:
	l=toeplitz_blockschur(input_f[:sizen*(meff+pad2*meff),:sizen*(meff+pad2*meff)],(meff+pad2*meff),1)
	print l[:,l.shape[1]-2:l.shape[1]]