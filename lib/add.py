###########################################
# Example f77 code (hello.f)
# 
#c###########################################################
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
#c###########################################################
#      subroutine arrfunc(x, y, z)
#      real*8 x(*)
#      real*8 y(*)
#      real*8 z
#      write (*,*) 'hello! this is a function'
#      y(1) = x(1)
#      y(2) = x(2)
#      y(3) = x(3)
#      z = y(1)
#      end
#c###########################################################
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
# part 1
x = np.array(np.arange(6)) + 0.5
print(x)
y = ctypes.c_int(len(x))
x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
z = ctypes.c_double(20.0)
myLibx.func_(x, ctypes.byref(y), ctypes.byref(z))
print(x)
print(y)
print(z)
print('############################')

# part 2 : 
# basically, the idea is to pass arrays as they are (because we pass the array
# pointers), while passing singleton variables using ctypes.byref
x = np.array(np.arange(6)) + 0.5
xref = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
y = np.array(np.arange(6)) + 2.5
yref = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
z = ctypes.c_double(20.0)
myLibx.arrfunc_(xref,yref, ctypes.byref(z))
print(x); print(y); print(z)
print('############################')

# part 3
x = ctypes.c_double(5.0);
y = ctypes.c_double(11.0);
z = ctypes.c_double(1.0);
myLibx.addfunc_(ctypes.byref(x), ctypes.byref(y), ctypes.byref(z))
print(z)
print('############################')

# part 4
x = ctypes.c_int(5);
y = ctypes.c_double(11.0);
z = ctypes.c_double(1.0);
myLibx.test1_(ctypes.byref(x), ctypes.byref(y), ctypes.byref(z))
print(z)
x = ctypes.c_int(-1);
y = ctypes.c_double(11.0);
z = ctypes.c_double(1.0);
myLibx.test1_(ctypes.byref(x), ctypes.byref(y), ctypes.byref(z))
print(z)



