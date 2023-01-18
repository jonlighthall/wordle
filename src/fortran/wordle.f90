! beat wordle
program wordle
  implicit none
  integer(kind=4) :: read_unit,ierr,i,nlines = 0,ilet,idx
  integer,parameter :: wl=5
  integer(kind=4),dimension(wl) :: max_count,max_let,max_char
  character(len=wl) :: line,mode
  integer,dimension(wl,26) :: counts = 0

  ! read dictionary file of 5-letter words
  open(newunit=read_unit,file='words_alpha5.txt',status='old',action='read')
  do
     read(read_unit,'(a)',iostat=ierr) line
     if (ierr.eq.0) then
        if (mod(nlines,1000).eq.0) then
           print*,line
        endif
        do i=1,wl
           ilet=ichar(line(i:i),kind(ilet))
           counts(i,ilet-96) = counts(i,ilet-96) + 1
        enddo
        nlines = nlines + 1
     elseif (ierr<0) then
        print'(/a)','EOF'
        print 1,'      number of lines = ',nlines
1       format(a,i7)
        do i=1,wl
           max_count(i)=maxval(counts(i,:))
           max_let(i)=maxloc(counts(i,:),1)
           max_char(i)=max_let(i)+96
           mode(i:i)=char(max_char(i))
           print '(1x,i4,1x,i2,1x,i3,1x,a)',max_count(i),max_let(i),max_char(i),mode(i:i)
        enddo
        print*,mode
        exit
     endif
  enddo
  idx=maxloc(max_count,1)
  print '(3a,i1)','the most likely letter is ''',mode(idx:idx),''' in position ',idx
  close(read_unit)
end program wordle
