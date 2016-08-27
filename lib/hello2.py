# junk code ignore

import ctypes 

myLibx = ctypes.cdll.LoadLibrary('./libhello2.so') # this is a bit of a pain. 
                                    # unless a new python console is started
x = ctypes.c_double(5);
y = ctypes.c_double(11);
myLibx.test2_(ctypes.byref(x), ctypes.byref(y))
print(x)
print(y)





