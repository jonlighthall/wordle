program filter
  implicit none
  integer(kind=4) :: read_unit,ierr
  character(len=64) :: line
  integer :: nlines = 0
  integer :: length,mxlen = 0

  open(newunit=read_unit,file='words_alpha.txt',iostat=ierr,status='old',action='read')
  print*,'iostat = ',ierr
  if (ierr==0) then
     print*,'OK'
  else
     print*,'ERROR'
  endif
  do
     read(read_unit,'(a)',iostat=ierr) line
     length = len(trim(line))
     if (length.gt.mxlen) then
        mxlen=length
     endif
     nlines = nlines + 1
     if (mod(nlines,1000)==0) then
        print*,line
     endif     
     if (ierr<0) then
        print*,'EOF'
        print*,'number of lines = ',nlines
        print*,'longest word length = ',mxlen
        exit
     endif
  enddo
  close(read_unit)
end program filter
