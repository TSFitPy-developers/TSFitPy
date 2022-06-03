program faltbo

! This version can read up to 100 columns and convolve them all.
! The convolution window adapts to the wavelength
! BPz 13/02-2019

  implicit none

  integer                                            :: iprf,ii,jj,ios,isize,isize2,jmax,i
  integer                                            :: resample,itype,ipad,jbeg,jend,ipad2,ipad1,kk
  real(kind=kind(1.d0))                              :: FWHM,FWHM2,somme,dlam,dlam2,clight
  real(kind=kind(1.d0)), allocatable, dimension(:)   :: lambda
  real(kind=kind(1.d0)), dimension(102)              :: Ftest
  real(kind=kind(1.d0)), allocatable, dimension(:)   :: lambda2,F2,dl,prof
  real(kind=kind(1.d0)), allocatable, dimension(:,:) :: Fconv,Farray
  real(kind=kind(1.d0)), parameter                   :: pi = 4.*atan(1.)
  character(len=800)                                 :: inspec,outfil,command,oneline
  logical                                            :: velocity

  clight = 2.99792458e5 
  resample=1
  ! read inputs and open files 
  print*,'input file ?'
  read(5,' (A) ') inspec
  print*,'output file ?'
  read(5,' (A) ') outfil 

  ios=0
  open(unit=10,file=inspec,status='old',iostat=ios,action='read')
  if (ios==0) print*,'input file opened'
  if (ios/=0) stop "problem in file opening ..."
  open(unit=20,file=outfil,status='unknown')

  print*,'FWHM (in km/s or mA, <0 if velocity) ?'
  read(5,*) FWHM
  print*,'profile type ? (1 = exp - 2 = gauss - 3 = rad.-tan. - 4 = rot.)'
  read(5,*) iprf         ! profile type
!
! we don't give the choice anymore.
! we suppose that spectra are always 2 or 3 or 4 or 5 columns: 1st is
! lambda, the next 1 or 2 is/are Fnorm and/or Fabs, and the 4th is 
! absolute intensity, the 5th is normalized intensity
! we convolve all. BPz 6/02-2019
!
!  print*,'1 = convol. Fnorm; 2 = convol. Fabs'
!  read(5,*) itype        ! 1 if convol. of Fnorm, 2 if convol. of Fabs
!  print*,'step (resampling) for output (< 100)?'
!  read(5,*) resample     ! nb of wavelengths to be skipped in output (max. 99)

  velocity=.false.
  if (FWHM.LT.0.) velocity=.true.

! assumes no headers in inspec ... 

!  command='wc -l ' // inspec
!  call system(command)

! read input spectrum 

! first determine number of columns in input (read a dummy line first to deal with a possible header)
  read(unit=10, fmt=*)
  ios=0
  read(unit=10,iostat=ios,fmt='(a)') oneline
  do jmax=1,102
    read(oneline,iostat=ios,fmt=*) Ftest(1:jmax)
    if (ios /= 0) exit
  enddo
  jmax=jmax-2
  if (jmax == 100) then
    stop ' input file may have more than 100 columns, increase jmax!'
  endif
  print*,jmax,' columns to convolve in input'
  backspace(10)
! and number of lines in input file
  isize=0
  ios=0
  do while (ios == 0)
    read(unit=10,fmt=*,iostat=ios) Ftest(1)
    isize=isize+1
  enddo
  isize=isize-1
  print*,isize,' lines to read '
  rewind(10)

  allocate(Farray(isize,jmax))
  allocate(lambda(isize))

! now read
  ios=0
  do ii=1,isize
    read(unit=10,iostat=ios,fmt=*) lambda(ii),Farray(ii,1:jmax)
  end do
  if (ios /= 0) then
    stop 'something went wrong while reading'
  endif

  print*,isize,' lines read in input file'
  allocate(Fconv(isize,jmax))

  ! define how many 0s necessary before and after and extend input spectrum 
  dlam=lambda(2)-lambda(1)
  dlam2=lambda(isize)-lambda(isize-1)
  if (velocity) then
     ipad1=int(-50.*FWHM*lambda(1)/clight/dlam)+1
     ipad2=int(-50.*FWHM*lambda(isize)/clight/dlam2)+1
  else
     ipad1=int(50.*FWHM/1.e3/dlam)+1
     ipad2=int(50.*FWHM/1.e3/dlam2)+1
  endif
  print*,'ipad1, ipad2, isize ',ipad1,ipad2,isize
  isize2=isize+ipad1+ipad2
  print*,'ipad1+ipad2, isize2 = ',ipad1+ipad2,isize2 
  allocate(lambda2(isize2),F2(isize2))
  do ii=1,ipad1
     lambda2(ii)=lambda(1)-dlam*(ipad1-ii+1)
     F2(ii)=0.
  end do
  do ii=1,ipad2
     lambda2(isize+ipad1+ii)=lambda(isize)+dlam2*ii
     F2(isize+ipad1+ii)=0.
  end do

! loop to convolve 4 spectra (if needed)
  i=0
  do itype=1,jmax
    do ii=1,isize
     kk=ii+ipad1
     lambda2(kk)=lambda(ii)
     F2(kk)=Farray(ii,itype)
    end do

    FWHM2=FWHM/1.e3
    do ii=1,isize,resample
     ! at current wavelength: check step, and allocate conv. profile accordingly
     if (ii.eq.isize) then
       dlam=lambda(ii)-lambda(ii-1)
     else
       dlam=lambda(ii+1)-lambda(ii)
     endif
     if (velocity) then
       FWHM2=-FWHM*lambda(ii)/clight ! km/s -> A 
       ipad=int(-50.*FWHM*lambda(ii)/clight/dlam)+1
     else
       ipad=int(50.*FWHM/1.e3/dlam)+1
     endif

     jbeg=1
     jend=2*ipad+1
     kk=ii+ipad1
     allocate(dl(jend))
     allocate(prof(jend))
     do jj=jbeg,jend
        dl(jj)=lambda2(kk+jj-1-ipad)-lambda2(kk)  !where segfault is occurring
     end do
     ! call relevant procedure for convolution profile
     if (iprf==1 .or. iprf==2) then
       call gauss(ipad,dl,iprf,FWHM2,prof,jbeg,jend)
     else if (iprf==3) then
       call radtan(ipad,dl,FWHM2,prof,jbeg,jend)
     else if (iprf==4) then
       call rota(ipad,dl,FWHM2,lambda(ii),prof,jbeg,jend)
     else 
       stop 'undefined profile'
     endif
     ! if (iprf==5) call FTS(FWHM)

     ! normalization of convolution profile 
     ! somme=0.
     somme=prof(jbeg)*(dl(jbeg+1)-dl(jbeg))*0.5
     do jj=jbeg+1,jend-1
        ! somme=somme+prof(jj)
        somme=somme+prof(jj)*(dl(jj+1)-dl(jj-1))*0.5
     end do
     somme=somme+prof(jend)*(dl(jend)-dl(jend-1))*0.5
     if (somme.le.0) stop 'problem with profile normalization ...'
     do jj=jbeg,jend
        prof(jj)=prof(jj)/somme
     end do

     ! convolution : f'(ii) = sum (C(j)*f(j))    j = ii-ipad -> ii+ipad 
     ! somme=0.
     somme=prof(jbeg)*F2(kk+jbeg-1-ipad)*(dl(jbeg+1)-dl(jbeg))*0.5
     do jj=jbeg+1,jend-1
       ! somme=somme+prof(jj)*F2(kk+jj-1-ipad)
       somme=somme+prof(jj)*F2(kk+jj-1-ipad)*(dl(jj+1)-dl(jj-1))*0.5
     end do
     somme=somme+prof(jend)*F2(kk+jend-1-ipad)*(dl(jend)-dl(jend-1))*0.5
     Fconv(ii,itype)=somme
     deallocate(dl)
     deallocate(prof)
    end do
    print*,'test5'
  enddo

!  print*,'convolution done ! Now writing output ...'

  ! output, with resampling 
  do ii=1,isize,resample
    write(20,200) lambda(ii),Fconv(ii,1:jmax)
  end do

200  format(f13.4,100(1x,1pe12.5))


end program faltbo

! ***************************************************************************

subroutine gauss(ipad,dl,iprf,FWHM,prof,jbeg,jend)

  ! exponential and gaussian profiles

  implicit none

  integer                                   :: ipad,iprf,i,jbeg,jend
  real(kind=kind(1.d0)),dimension(2*ipad+1) :: prof,dl
  real(kind=kind(1.d0))                     :: FWHM,const,y

  if (iprf==1) const=1.38629/FWHM
  if (iprf==2) const=1.66511/FWHM

! we require at least the 3 central points of the profile.
  jbeg=ipad
  jend=ipad+2

  do i=1,2*ipad+1
     prof(i)=0.
     y=dl(i)*const
     if (iprf==1) then
       if (exp(-abs(y)).gt.1.e-6) then
         prof(i)=exp(-abs(y))
         jbeg=min(jbeg,i)
         jend=max(jend,i)
       endif
     else if (iprf==2) then
       if (exp(-y*y).gt.1.e-6) then
         prof(i)=exp(-y*y)
         jbeg=min(jbeg,i)
         jend=max(jend,i)
       endif
     endif
  end do

  return

end subroutine gauss

! ***************************************************************************

subroutine radtan(ipad,dl,FWHM,prof,jbeg,jend)

  ! radial-tangential profile; macroturbulent velocity = 1.433*FWHM
  ! cf. Gray 1978, Solar Phys. 59, 193

  implicit none 

  integer                                       :: i,j,ipad,jbeg,jend
  real(kind=kind(1.d0)),dimension(2*ipad+1)     :: prof,dl
  real(kind=kind(1.d0))                         :: FWHM,width,delta,di,pp
  real(kind=kind(1.d0)),dimension(40),parameter ::  rtf = & 
       (/1.128,.939,.773,.628,.504,.399,.312,.240,.182,.133,   &
       .101,.070,.052,.037,.024,.017,.012,.010,.009,.007,.006, &
       .005,.004,.004,.003,.003,.002,.002,.002,.002,.001,.001, & 
       .001,.001,.001,.001,.000,.000,.000,.000 /)

! delta is the wavelength distance between given RTF points. 
  width=FWHM*1.433
  delta=width/10.

  do j=1,2*ipad+1
     prof(j)=0.
  end do

  prof(ipad+1)=rtf(1)

  j=ipad+2
  di=delta
  do i=2,35
     di=delta*float(i-1)
     do while (dl(j).le.di)
        pp=log(rtf(i))+(log(rtf(i-1))-log(rtf(i)))*(di-dl(j))/delta
        prof(j)=exp(pp)
        jend=j
        j=j+1
     end do
  end do

  j=ipad
  do i=2,36
     di=delta*float(i-1)
     do while (abs(dl(j)).le.di)
        pp=log(rtf(i))+(log(rtf(i-1))-log(rtf(i)))*(di-abs(dl(j)))/delta
        prof(j)=exp(pp)
        jbeg=j
        j=j-1
     end do
  end do

  return

end subroutine radtan

! ***************************************************************************

subroutine rota(ipad,dl,FWHM,lambda,prof,jbeg,jend)

  ! rotational profile; eps is wavelength dependant linear (in mu)
  ! limb-darkening coefficient found for F V -K IV stars (improve !!!)
  ! FWHM is v*sin i in wavelength units 

  implicit none 

  integer                                   :: i,ipad,jprof,nmax,jbeg,jend
  real(kind=kind(1.d0))                     :: FWHM,dlam,dlambda,lambda,eps
  real(kind=kind(1.d0)),dimension(2*ipad+1) :: prof,x
  real(kind=kind(1.d0)),dimension(2*ipad)   :: dl
  real(kind=kind(1.d0)),parameter           :: pi = 4.*atan(1.)

  eps=1.-0.3*lambda/5000.
  dlambda=FWHM
  jbeg=2*ipad+1
  jend=1
  do i=1,2*ipad+1
     if (abs(dl(i)).le.dlambda) then
        jbeg=min(jbeg,i)
        jend=max(jend,i)
        prof(i)=(2.*(1.-eps)*sqrt(1.-(dl(i)/dlambda)**2)+pi*0.5*eps* &
          (1.-(dl(i)/dlambda)**2))/(pi*dlambda*(1.-eps/3.))
     endif
  end do

  return

end subroutine rota

! ***************************************************************************

subroutine FTS(FWHM)

  ! FTS profile (sin(x)/x) - from Kurucz- program broaden.f
! obsviously not ready for use.!!

  implicit none 

  real(kind=kind(1.d0)) :: FWHM ! assumed in km/s ...

  ! x=(i-1.)*vstep/FWHM*2.*1.8954942
  ! red(i)=sin(x)/x*exp(-0.06*x**2)

  return

end subroutine FTS
