#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot

if len(sys.argv) < 1:
    print "Usage: %s lag.dat" % sys.argv[0] 
    sys.exit(1)

# Reconstruct fortran complex array in python
spec = np.fromfile(sys.argv[1],dtype='<f',count=-1)
n = len(spec)
print n
spec = np.reshape(spec,[n,1],order='F')
np.save('spec_real_data.npy',spec)
