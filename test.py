#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from mpi4py import MPI

a=np.array([1, 1-2j])

p=toeplitz(a)
print p