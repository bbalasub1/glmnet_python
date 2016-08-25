###########################################
# Example f77 code (hello.f)
# 
#      subroutine func(x, N, y)
#c
#11111 real*8 x(*)
#11113 real*8 y
#      integer N
#      integer i
#c
#c
#      y = 0
#      do 11120 i = 1, N, 1
#      y = y + x(i)
#11120 continue
#c 
#      end
#      
###########################################
# 
# Compile code:
#    gfortran hello.f -fPIC -shared -o libhello.so
#
###########################################
#   
# Note: compilation also works for GLMnet.f
#      
      
import numpy as np
import ctypes 

myLibx = ctypes.cdll.LoadLibrary('./libhello.so') # this is a bit of a pain. 
                                    # unless a new python console is started
                                    # the shared library will persist in memory
n = 5;
AType = ctypes.c_double*n
x = AType(1.1, 2.2, 3.3, 4.4, 5.5)

x = np.array(np.arange(6)) + 0.5
print(x)
y = ctypes.c_int(len(x))
x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
z = ctypes.c_double(20.0)
myLibx.func_(x, ctypes.byref(y), ctypes.byref(z))

print(x)
print(y)
print(z)
