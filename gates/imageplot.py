import sys
import numpy as np
import matplotlib.pyplot as plt
import os

    
np.set_printoptions(precision=2, suppress=True, linewidth=200)
filename='%s/Gate%s/%s' %(os.path.dirname(os.path.abspath(__file__)),sys.argv[2],sys.argv[1])
print filename
a = np.memmap(filename, dtype='float32', mode='r', shape=(16384,660),order='F')
plt.subplot(1, 1, 1)
plt.imshow(a.T, interpolation='nearest')
plt.colorbar()
plt.show()