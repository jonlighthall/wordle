! beat wordle
program wordle
  implicit none
  integer(kind=4) :: read_unit,ierr,i,nlines = 0
  integer(kind=4),dimension(5) :: write_unit
  integer(kind=8),dimension(5) :: prob=0
  character(len=5) :: line

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
           prob(i) = ichar(line(i:i),kind(prob(i)))
           if (mod(nlines,1000).eq.0) then
              print*,i,' = ',prob(i)
           endif
           if ((prob(i).lt.97).or.(prob(i).gt.122)) then
              print*,line
           endif
        enddo
        nlines = nlines + 1
     elseif (ierr<0) then
        print'(/a)','EOF'
        print 1,'      number of lines = ',nlines
1       format(a,i7)
        exit
     endif
  enddo
  close(read_unit)
end program wordle
