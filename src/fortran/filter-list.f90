! read a dictionary file and filter for length 5

program filter
  implicit none
  integer(kind=4) :: read_unit,ierr,write_unit
  character(len=64) :: line
  integer :: nlines = 0, n5 = 0
  integer :: length,mxlen = 0

  open(newunit=read_unit,file='words_alpha.txt',iostat=ierr,status='old',action='read')
  print *,'iostat = ',ierr
  if (ierr==0) then
     print*,'OK'
  else
     print*,'ERROR'
  endif
  open(newunit=write_unit,file='words_alpha5.txt')
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
     if (length.eq.5) then
        n5 = n5 +1
        write(write_unit,'(a)') line
     endif
     if (ierr<0) then
        print'(/a)','EOF'
        print 1,'    number of lines = ',nlines
        print 1,'longest word length = ',mxlen
1       format(a,i7)
        exit
     endif
  enddo
  close(read_unit)
  close(write_unit)
end program filter
