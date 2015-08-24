import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from mpi4py import MPI
#import h5py 
import time
from sp import multirate
from parallel_decom_pading import *

l=block_toeplitz_par("real_n_8_b_2_neff_16_beff_4.dat",16,4,0)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
	print l