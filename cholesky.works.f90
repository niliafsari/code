integer, parameter :: n=55
real, dimension(n,n) :: a,b,l

call random_seed
call random_number(a)
a=matmul(a,transpose(a))
b=a
call cholesky(a,n)
l=a
call tinv(l,n)
!write(*,*) b,l,b-matmul(a,transpose(a)),matmul(a,l)
a=matmul(transpose(l),l)
l=matmul(b,a)
write(*,*) l(n/2+1,:)
end

recursive subroutine cholesky(a,n)
! use lower triangle of a
real, dimension(n,n) :: a
real, dimension(n/2,n/2) :: a11
real, dimension(n-n/2,n-n/2) :: a22
real, dimension(n-n/2,n/2) :: a21

if (n .eq. 1) then
	a=sqrt(a)
	return
endif
n1=n/2
n2=n/2+1
nd=n-n/2
a11=a(:n1,:n1)
a21=a(n2:,:n1)
a22=a(n2:,n2:)
call cholesky(a11,n1)
a(:n1,:n1)=a11
call tinv(a11,n1)
a21=matmul(a21,transpose(a11))
a22=a22-matmul(a21,transpose(a21))
call cholesky(a22,nd)
a(n2:,n2:)=a22
a(:n1,n2:)=0
a(n2:,:n1)=a21
end subroutine


recursive subroutine tinv(a,n)
! inverse of lower triangular matrix
real, dimension(n,n) :: a
real, dimension(n/2,n/2) :: a11
real, dimension(n-n/2,n-n/2) :: a22
real, dimension(n-n/2,n/2) :: a21

if (n .eq. 1) then
	a=1/a
	return
endif
n1=n/2
n2=n/2+1
nd=n-n/2
a11=a(:n1,:n1)
a21=a(n2:,:n1)
a22=a(n2:,n2:)
call tinv(a11,n1)
call tinv(a22,nd)
a21=-matmul(matmul(a22,a21),a11)
a(:n1,:n1)=a11
a(n2:,:n1)=a21
a(n2:,n2:)=a22
end subroutine
