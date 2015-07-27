integer, parameter :: ndim=2,nf=16384,nt=1024,nred=1,nredt=2,ntin=1321/2,ns=2
real*4, dimension(nf/nred,nt) :: rawred
character*1 fn1
complex, dimension(nf/nred/2*ns) :: ctmp
complex, dimension(nf/nred*ns) :: cfact
real, dimension(nf/nred*ns) :: rfact
complex, dimension(nt,nf/nred/2) :: cfact2
real, dimension(nt,nf/nred/2) :: rfact2

rawred=0
open(10,file='raw4.dat',access='stream')
read(10) rawred(:,:ntin)
!call pmap('dspect.pgm',rawred,nf/nred,ntin,1)
cfact2=0
open(10,file='cfact2.dat',access='stream')
!$omp parallel do default(shared) private(cfact,ctmp,rfact),schedule(dynamic,1)
do i=1,ntin
!do j=1,ns
!rfact(j::ns)=rawred(:,i)
!enddo
rfact=0
rfact(:nf/nred)=rawred(:,i)
rfact=rfact*(count(rfact.ne.0))/sum(rfact)
if (any(rfact>100)) rfact=0
where(rfact.eq.0) rfact=1
rfact=max(0.0001,rfact)
cfact=sqrt(rfact)
!do j=1,64
!rfact=(2*rfact+cshift(rfact,1)+cshift(rfact,-1))/4
!enddo
!rfact=max(0.02,rfact)
!cfact=rfact
call four1(cfact,nf/nred,-1)
cfact(3*nf/nred/2+1:)=cfact(nf/nred/2+1:nf/nred)
cfact(nf/nred/2+1:3*nf/nred/2)=0
call four1(cfact,nf/nred*ns,1)
cfact=abs(cfact)**2
call four1(cfact,nf/nred*ns,-1)
if (count(rawred(:,i) .ne. 0) .eq. 0) cycle
call toeplitz_cholesky(nf*ns/nred/2,ctmp,cfact)
!cfact2(i,:)=cfact(:nf/nred/2)!ctmp
cfact2(i,:)=ctmp(:nf/nred/2)
enddo
do i=1,nf/nred/2
call four1(cfact2(1,i),nt,1)
enddo
write(10) cfact2
cfact2(:,1)=0
rfact2=cshift(abs(cfact2),nt/2)
call pmap('rfact2.pgm',rfact2,nt,nf/nred/2,2)


contains
  subroutine pmap(fn,rmap1,nx,ny,iscale0)
  real rmap(nx,ny),rmap1(nx,ny)
  integer*2, dimension(nx,ny) :: imap
  integer*1, dimension(nx,ny) :: imap1
  character(len=*):: fn
  integer npix,mypos

  npix=min(ny/2-1,nx/2-1,300)
  iscale=iscale0
  
  rmap=rmap1
  do while (iscale > 1)      
     rmap=sign((sqrt(abs(rmap))),rmap)
     iscale=iscale-1
  end do
  rmax=maxval(rmap)
  rmin=minval(rmap)
  write(*,*) trim(fn),rmax,rmin
  imap=255*(rmap-rmin)/(rmax-rmin)
  imap1=127*(rmap-rmin)/(rmax-rmin)
  open(10,file=fn)
  write(10,'(2hP5)')
  write(10,*)nx,ny
  write(10,*) 255
!  write(10,*) 127
  INQUIRE(UNIT=10, POS=mypos)
  close(10)
  open(10,file=fn, access='stream',position='append')
!  write(10,pos=mypos) int(imap,1)
  write(10) int(imap,1)
  close(10)
end subroutine pmap

end

recursive subroutine toeplitz_cholesky(n,out,in)
  complex, dimension(n) :: in
  complex, dimension(n) :: out
  complex, dimension(n) :: alpha,beta,beta0
  complex gamma
  
  alpha(:n-1)=-conjg(in(2:))
  beta=conjg(in)
  alpha(n)=0
  out=0
  do i=1,n*10
     if (real(beta(1)) .le. 0) then
        write(*,*) i,beta(1) , 'not +ve def'
	out=0
	return
     end if
     s = sqrt(real(beta(1)))
     gamma=alpha(1)/real(beta(1))
     beta0=beta
     beta=beta-conjg(gamma)*alpha
     alpha=eoshift(alpha-gamma*beta0,1)     
  end do
  do i=1,n
     if (real(beta(1)) .le. 0) then
        write(*,*) i,beta(1) , 'not +ve def'
	out=0
	return
     end if
     s = sqrt(real(beta(1)))
     out(i)=conjg(beta(n-i+1))/s
     gamma=alpha(1)/real(beta(1))
     beta0=beta
     beta=beta-conjg(gamma)*alpha
     alpha=eoshift(alpha-gamma*beta0,1)     
  end do
  out=out(n:1:-1)
end subroutine toeplitz_cholesky

