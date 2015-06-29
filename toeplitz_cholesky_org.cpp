#include <vector>
#include <cassert>
#include <cstring>
#include <complex>
#include <iostream>
#include <stdexcept>

using namespace std;


//
// Given Hermitian Toeplitz matrix A, compute the lower triangular
// Cholesky factor L, such that A = L L^T.
//
//   @n = size of matrix
//   @out = array of length n^2, will contain L on exit
//   @in = array of length n, contains the first column of A
//
void toeplitz_cholesky(int n, complex<double> *out, const complex<double> *in)
{
    assert(n >= 2);      // I didn't bother getting the n=1 case correct
    memset(out, 0, n*n * sizeof(complex<double>));

    vector<complex<double> > alpha(n+1);
    for (int i = 0; i < n-1; i++)
	alpha[i] = -conj(in[i+1]);

    vector<complex<double> > beta(n+1);
    for (int i = 0; i < n; i++)
	beta[i] = conj(in[i]);

    // These sentinel values will always be zero, but simplify the logic below
    alpha[n-1] = complex<double>(0,0);
    alpha[n] = complex<double>(0,0);
    beta[n] = complex<double>(0,0);

    for (int i = 0; i < n; i++) {
	if (beta[0].real() <= 0.0)
	    throw runtime_error("Fatal: matrix in toeplitz_cholesky() is not positive definite, Cholesky factorization failed");

	// Fill i-th column of LD
	double s = sqrt(beta[0].real());
	for (int j = i; j < n; j++)
	    out[j*n+i] = conj(beta[j-i]) / s;
	
	// Update alpha, beta arrays
	complex<double> gamma = alpha[0] / beta[0].real();
	cout <<"gamma="<< gamma << endl;
	for (int j = 0; j < n; j++) {
	    beta[j] -= conj(gamma) * alpha[j];
	    alpha[j] = alpha[j+1] - gamma * beta[j+1];
	}
	if (i==0) for (int j=0;j<n;j++) cout << alpha[j];
    }
}


int main(int argc, char **argv)
{
    // An arbitrarily chosen example matrix
    const int n = 2;
    complex<double> a[n] = { complex<double>(10, 10), 
			     complex<double>(1, -2) };

    complex<double> l[n*n];
    toeplitz_cholesky(n, l, a);
    
#if 1  // set to 1 to inspect the matrix
    cout << "Cholesky factor follows\n";
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++)
	    cout << " " << l[i*n+j];
	cout << "\n";
    }
#endif

    //
    // End-to-end check on the factorization follows...
    //

    // Check lower triangular
    for (int i = 0; i < n; i++)
	for (int j = i+1; j < n; j++)
	    assert(l[i*n+j] == complex<double>(0,0));

    // Compute L L^dag
    complex<double> ll[n*n];
    memset(ll, 0, n*n * sizeof(complex<double>));
    for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	    for (int k = 0; k < n; k++)
		ll[i*n+j] += l[i*n+k] * conj(l[j*n+k]);

    // Compare L L^dag to input matrix A
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    complex<double> aij = (i >= j) ? a[i-j] : conj(a[j-i]);
	    num += norm(aij - ll[i*n+j]);
	    den += norm(aij);
	}
    }
    cout << "End-to-end check: the following number should be small: " << sqrt(num/den) << endl;

    return 0;
}
