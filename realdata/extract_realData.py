import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from mpi4py import MPI
#import h5py 
import time
from sp import multirate
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 6:
    print "Usage: %s filename num_rows num_columns n b" % (sys.argv[0])
    sys.exit(1)

a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(int(sys.argv[2]),int(sys.argv[3])),order='F')
plt.subplot(1, 1, 1)
plt.plot(a)
#plt.xlabel('number of threads')
plt.show()
n=int(sys.argv[4])
b=int(sys.argv[5])
pad=1
a_input=np.zeros(shape=(n+n*pad, b+b*pad), dtype=complex)
a_input[:n,:b]=np.copy(a[1000:1000+n,500:500+b])
del a
neff=n+n*pcd ad
beff=b+b*pad
pad2=1
input_f=np.zeros(shape=(beff+pad2*beff, neff*(beff+pad2*beff)), dtype=complex)
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
for i in xrange(0,neff):
	input_f[:(beff+pad2*beff),i*(beff+pad2*beff):(i+1)*(beff+pad2*beff)]=toeplitz(np.append(a_input[i,:],np.zeros(pad2*beff)))
output_file="real_n_%s_b_%s_neff_%s_beff_%s.dat" %(str(n),str(b),str(neff),str(beff+pad2*beff))
output = np.memmap(output_file, dtype='complex', mode='w+', shape=(beff+pad2*beff, neff*(beff+pad2*beff)),order='F')
output[:]=input_f[:]
del output