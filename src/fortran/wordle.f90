! beat wordle
program wordle
  implicit none
  integer(kind=4) :: read_unit,ierr,i,nlines = 0,ilet
  integer(kind=4),dimension(5) :: write_unit,iprob = 0
  integer(kind=8),dimension(5) :: isum = 0
  real,dimension(5) :: rprob = 0.
  character(len=5) :: line,word

  ! read dictionary file of 5-letter words
  open(newunit=read_unit,file='words_alpha5.txt',iostat=ierr,status='old',action='read')
  print *,'iostat = ',ierr
  if (ierr==0) then
     print*,'OK'
  else
     print*,'ERROR'
  endif
  do
     read(read_unit,'(a)',iostat=ierr) line
     if (ierr.eq.0) then
        if (mod(nlines,1000).eq.0) then
           print*,line
        endif
        do i=1,5
           ilet=ichar(line(i:i),kind(ilet))
           if ((ilet.lt.97).or.(ilet.gt.122)) then
              print*,line
           endif
           isum(i) = isum(i) + int(ilet,kind(isum(i)))
           ! no, you need 26x5 array and take max value
           if (mod(nlines,1000).eq.0) then
              print*,i,' = ',ilet
           endif
        enddo
        nlines = nlines + 1
     elseif (ierr<0) then
        print'(/a)','EOF'
        print 1,'      number of lines = ',nlines
1       format(a,i7)
        do i=1,5
           rprob(i) = real(isum(i))/real(nlines)
           iprob(i) = nint(rprob(i))
           word(i:i) = char(iprob(i))
        enddo
        print*,word
        exit
     endif
  enddo
  close(read_unit)
end program wordle
