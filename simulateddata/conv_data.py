#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot

if len(sys.argv) < 1:
    print "Usage: %s lag.dat" % sys.argv[0] 
    sys.exit(1)

# Reconstruct fortran complex array in python
lag = np.fromfile(sys.argv[1],dtype='<f',count=-1)
n = len(lag) / 2
print n
# The data is loaded as a 1-D array, with no dimensional information
# The 2n-1 components are real, and the 2n components are complex
# Break into complex and real, then assign them explicitly in A
lag = np.reshape(lag,[n,2])
A = np.zeros(n, dtype=np.complex)
A.real = lag[:,0]
A.imag = lag[:,1]
A = np.reshape(A,[512,1024],order='F')

# A good sanity check: A[0,:] should be entirely real, otherwise the resulting matrices will not be positive definite.

#print A[0:10][0:10]

np.save('simdata.npy',A)


