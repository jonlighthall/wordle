! beat wordle
program wordle
  implicit none
  integer(kind=4) :: read_unit,ierr,i,nlines = 0,ilet,idx
  integer,parameter :: wl=5 ! word length
  integer(kind=4),dimension(wl) :: max_count,max_let,max_char
  character(len=wl) :: line,mode
  integer,dimension(wl,26) :: counts = 0

  ! read dictionary file of 5-letter words
  open(newunit=read_unit,file='data/words_alpha5.txt',status='old',action='read')
  do
     read(read_unit,'(a)',iostat=ierr) line
     if (ierr.eq.0) then
        ! print every Nth line
        if (mod(nlines,1000).eq.0) then
           print*,line
        endif
        ! scan through each word
        do i=1,wl
           ! extract letter
           ilet=ichar(line(i:i),kind(ilet))
           ! increment counter
           counts(i,ilet-96) = counts(i,ilet-96) + 1
        enddo
        nlines = nlines + 1
     elseif (ierr<0) then
        print'(/a)','EOF'
        print 1,'      number of lines = ',nlines
1       format(a,i7)
        print '(1x,a)','count pos ascii letter'
        ! loop over every position
        do i=1,wl
           ! get most-found letter
           max_count(i)=maxval(counts(i,:))
           max_let(i)=maxloc(counts(i,:),1)
           ! convert to ASCII
           max_char(i)=max_let(i)+96
           mode(i:i)=char(max_char(i))
           print '(1x,i5,1x,i3,1x,i5,1x,a)',max_count(i),max_let(i),max_char(i),mode(i:i)
        enddo
        print*,mode
        exit
     endif
  enddo
  idx=maxloc(max_count,1)
  print '(3a,i1)','the overall most likely letter is ''',mode(idx:idx),''' in position ',idx
  close(read_unit)

  ! calcualte probability for each position
  ! the sum of each positions's probability should be the same

  !then rewind and loop through the word list again
  ! assign each word a likelyhood probabilty based on the calculated probability fore each letter
  ! that is, sum the per-letter probability
  ! this will give you the most lieky word(s)

  ! then ask for user input
  ! recycle through the list matching letters and postions
  ! inputs 0,1,2
  ! eg 01120


end program wordle
