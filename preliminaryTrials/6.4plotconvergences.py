import numpy as np
import matplotlib.pyplot as plt
from pypalettes import load_palette
colors = load_palette("Apricot")

orders=np.linspace(2,10,9)
rmsxh01=np.array([2.83431392e-13, 2.78742071e-13, 2.78744457e-13, 2.78744004e-13, 2.78743168e-13, 2.78743168e-13, 2.78743168e-13, 2.78743168e-13, 2.78743168e-13])
stdrmsxh01=np.array([1.29231748e-13, 1.29959542e-13, 1.29937822e-13, 1.29937616e-13, 1.29936690e-13, 1.29936690e-13, 1.29936690e-13, 1.29936690e-13, 1.29936690e-13])

# rmsxh08=np.array()
# stdrmsxh08=np.array()

rmsxh1=np.array([5.09329580e-07,7.28402546e-08, 8.84989448e-09, 6.49769258e-10, 8.41810736e-11, 5.33244687e-11, 3.07666867e-11, 2.86895711e-11,2.86432182e-11])
stdrmsxh1=np.array([2.43115790e-08, 4.27852354e-09, 6.11021526e-10, 4.96921188e-11, 7.77078504e-12, 4.26545804e-12, 9.22958257e-13, 5.57187747e-13, 5.45444269e-13])

rmsxh1_5=np.array([6.77546479e-06, 1.40153230e-06, 1.05238155e-07, 9.73622852e-08, 6.31427989e-08, 2.06996777e-08, 2.43151020e-09, 3.67642206e-09, 3.14724416e-09])
stdrmsxh1_5=np.array([2.86821960e-08, 1.57330583e-08, 4.68977625e-09, 9.17835432e-10,9.88632695e-11, 2.70142168e-10, 7.58518994e-10, 1.05324991e-09,1.05936341e-09])

rmsxh2=np.array([3.13087153e-06, 6.53523964e-07, 4.65766209e-08, 4.87270881e-08, 3.08079071e-08, 1.01832688e-08, 5.04762765e-09, 5.76742778e-09, 5.39471401e-09])
stdrmsxh2=np.array([2.86413386e-07, 6.78412563e-08, 4.81481051e-09, 7.26455461e-09, 4.24069992e-09, 2.01759051e-09, 3.84758022e-09, 4.05846388e-09,3.88211134e-09])

plt.errorbar(orders,rmsxh2, yerr=stdrmsxh2,fmt="o", capsize=4, label="h=2/m", color=colors[0])
plt.fill_between(orders, rmsxh2-stdrmsxh2, rmsxh2+stdrmsxh2, alpha=0.1, color=colors[0])
plt.errorbar(orders,rmsxh1_5, yerr=stdrmsxh1_5,fmt="o", capsize=4, label="h=1.5/m", color=colors[1])
plt.fill_between(orders, rmsxh1_5-stdrmsxh1_5, rmsxh1_5+stdrmsxh1_5, alpha=0.1, color=colors[1])

plt.errorbar(orders,rmsxh1,  yerr=stdrmsxh1,  fmt="o-",  capsize=4, label="h=1/m", color=colors[2])
plt.fill_between(orders, rmsxh1-stdrmsxh1, rmsxh1+stdrmsxh1, alpha=0.1, color=colors[2])

plt.errorbar(orders,rmsxh01, yerr=stdrmsxh01,fmt="o-", capsize=4, label="h=0.01/m", color=colors[5])
plt.fill_between(orders, rmsxh01-stdrmsxh01, rmsxh01+stdrmsxh01, alpha=0.1, color=colors[5])

plt.yscale("log")
plt.xlabel("# terms in the expansion")
plt.ylabel("mean RMS x [m]")
plt.grid(True)
plt.legend()
plt.show()