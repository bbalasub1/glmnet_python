      subroutine func(x, N, y)
c
11111 real*8 x(*)
11113 real*8 y
      integer N
      integer i
c
c
      y = 0
      do 11120 i = 1, N, 1
      y = y + x(i)
11120 continue
c 
      end
