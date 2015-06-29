import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import inv, toeplitz
import matplotlib.pylab as plt
import matplotlib.cm as cm

#Schur's algoritm for decomposing a Toeplitz matrix
#Input a is the first column of the Toeplitz matrix to be decomposed
def toeplitz_decomp(a):
    n = len(a)
    l = np.zeros(shape=(n,n), dtype=complex)
    alpha=np.append(-np.conj(a[1:]),0)
    beta=np.conj(a)

    for i in xrange(n):
        if np.real(beta[0]) < 0:
            print("Loop: ",i)
            print("beta[0] = ",np.real(beta[0]))
            print("ERROR - not positive definite")
            break

        s = np.sqrt(np.real(beta[0]))
        l[i:n,i] = np.conj(beta)[0:n-i] / s

        gamma = alpha[0] / np.real(beta[0])
        beta0 = np.array(beta)
        beta -= np.conj(gamma)*alpha
        alpha -= gamma*beta0
        alpha = np.append(alpha[1:],0)

    return l

def toep_zpad(a,npad):
    n = len(a)
    l = np.zeros(shape=(n,n), dtype=complex)
    alpha=np.append(-np.conj(a[1:]),0)
    beta=np.conj(a)

    for i in xrange(npad*n):
        if np.real(beta[0]) < 0:
            print("Loop: ",i)
            print("beta[0] = ",np.real(beta[0]))
            print("ERROR - not positive definite")
            break

        gamma = alpha[0] / np.real(beta[0])
        beta0 = np.array(beta)
        beta -= np.conj(gamma)*alpha
        alpha -= gamma*beta0
        alpha = np.append(alpha[1:],0)

    for i in xrange(n):
        if np.real(beta[0]) < 0:
            print("Loop: ",i)
            print("beta[0] = ",np.real(beta[0]))
            print("ERROR - not positive definite")
            break

        s = np.sqrt(np.real(beta[0]))
        l[i:n,i] = np.conj(beta)[0:n-i] / s

        gamma = alpha[0] / np.real(beta[0])
        beta0 = np.array(beta)
        beta -= np.conj(gamma)*alpha
        alpha -= gamma*beta0
        alpha = np.append(alpha[1:],0)

    return l

#Schur's algoritm for decomposing a block Toeplitz matrix
#Input a is the first column of the block Toeplitz matrix to be decomposed

def block_toeplitz_decomp(a):
    n=a.shape[0]	
    b=a.shape[1]	
    #l = np.zeros(shape=(n,b,b), dtype=complex)
    l = np.zeros(shape=(n,n,b,b), dtype=complex)
    e = np.zeros(shape=(1,b,b), dtype=complex)

    alpha=np.concatenate( (-np.conj(a[1:,:,:]),e), axis=0 )
    beta=np.conj(a)

    for i in xrange(n):
        #s = toeplitz_decomp(np.real(beta[0,0,:]))
	#print np.array([np.real(beta[0,:,0])[0], beta[0,:,0][1]])
        s = toeplitz_decomp(np.array([np.real(beta[0,:,0])[0], beta[0,:,0][1]]))
        #l[i] = np.dot( np.conj(beta)[n-i-1] , inv(s) )
        l[i:n,i] = np.dot( np.conj(np.swapaxes(beta,1,2))[0:n-i] , inv(s) )

        gamma = np.dot( alpha[0] , inv(np.array([np.real(beta[0])[0], beta[0][1]])) )
        beta0 = 1*beta
        beta -= np.swapaxes( np.dot(np.conj(gamma.T), alpha), 0, 1)
        alpha -= np.swapaxes( np.dot(gamma, beta0), 0, 1 )
        alpha = np.concatenate( (alpha[1:,:,:],e), axis=0 )

    return l

def blocktoep_zpad(a,npad):
    n=a.shape[0]	
    b=a.shape[1]	
    #l = np.zeros(shape=(n,b,b), dtype=complex)
    l = np.zeros(shape=(n,n,b,b), dtype=complex)
    e = np.zeros(shape=(1,b,b), dtype=complex)

    alpha=np.concatenate( (-np.conj(a[1:,:,:]),e), axis=0 )
    beta=np.conj(a)

    for i in xrange(npad*n):
        gamma = np.dot( alpha[0] , inv(np.real(beta[0])) )
        beta0 = 1*beta
        beta -= np.swapaxes( np.dot(np.conj(gamma), alpha), 0, 1)
        alpha -= np.swapaxes( np.dot(gamma, beta0), 0, 1 )
        alpha = np.concatenate( (alpha[1:,:,:],e), axis=0 )

    for i in xrange(n):
        #s = toep_zpad(np.real(beta[0,0,:]),npad)
        s = toep_zpad(beta[0,:,0])
        #l[i] = np.dot( np.conj(beta)[n-i-1] , inv(s) )
        l[i:n,i] = np.dot( np.conj(beta)[0:n-i] , inv(s) )

        gamma = np.dot( alpha[0] , inv(np.real(beta[0])) )
        beta0 = 1*beta
        beta -= np.swapaxes( np.dot(np.conj(gamma), alpha), 0, 1)
        alpha -= np.swapaxes( np.dot(gamma, beta0), 0, 1 )
        alpha = np.concatenate( (alpha[1:,:,:],e), axis=0 )

    return l

def plot_results(A):
    A = np.abs(A)
    vmin = A.mean()-2.*A.std()
    vmax = A.mean()+5.*A.std()

    plt.imshow(A, aspect='auto', cmap=cm.Greys, interpolation='nearest', vmax=vmax, vmin=0, origin='lower')

    plt.colorbar()
    plt.ylabel("tau")
    plt.xlabel("fd")

    plt.show()
