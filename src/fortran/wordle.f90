! beat wordle
program wordle
  implicit none
  integer(kind=4) :: read_unit,ierr,i,nlines = 0,ilet,idx
  integer,parameter :: wl=5
  integer(kind=4),dimension(wl) :: write_unit,iprob = 0,max_count,max_let,max_char
  integer(kind=8),dimension(wl) :: isum = 0
  real,dimension(wl) :: rprob = 0.
  character(len=wl) :: line,mean,mode,word
  integer,dimension(wl,26) :: counts = 0

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
        do i=1,wl
           ilet=ichar(line(i:i),kind(ilet))
           if ((ilet.lt.97).or.(ilet.gt.122)) then
              print*,line
           endif
           counts(i,ilet-96) = counts(i,ilet-96) + 1
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
        do i=1,wl
           rprob(i) = real(isum(i))/real(nlines)
           iprob(i) = nint(rprob(i))
           mean(i:i) = char(iprob(i))

           !           print*,(counts(i,:))
           max_count(i)=maxval(counts(i,:))
           max_let(i)=maxloc(counts(i,:),1)
           max_char(i)=max_let(i)+96
           mode(i:i)=char(max_char(i))
           print*,max_count(i),max_let(i),max_char,mode(i:i)
        enddo
        print*,mean
        print*,mode
        exit
     endif
  enddo
  
  idx=maxloc(max_count,1)
  print*,'most likely letter is ',mode(idx:idx),' in position ',idx

  
  close(read_unit)
end program wordle
