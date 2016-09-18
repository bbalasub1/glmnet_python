import matplotlib.pyplot as plt
import scipy

N = 10
a = list()
s = list()
ns = list()
for i in range(10):
    x = scipy.random.normal(size = N)
    a.append(scipy.mean(x))
    s.append(scipy.var(x))
    ns.append(N)
    N = 2*N

plt.plot(ns, s)    

