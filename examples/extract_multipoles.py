import bpmeth
import numpy as np
import matplotlib.pyplot as plt


quad = bpmeth.FieldExpansion(b=(0, 0.2, 0.04))
xarr = np.linspace(-0.1, 0.1, 100)
yarr = np.linspace(-0.1, 0.1, 100)
sarr = np.linspace(-0.3, 0.3, 100)
quad_fm = quad.create_fieldmap(xarr, yarr, sarr)

fig, ax = plt.subplots()
quad_fm.s_multipoles(3, ax=ax)
quad_fm.s_multipoles(3, ax=ax, method="fft")
quad_fm.s_multipoles(3, ax=ax, method="finite_difference")
plt.legend(fontsize=13)
ax.set_xlabel("s [m]", fontsize=13)
ax.set_ylabel(r"Multipole strength [1/m$^n$]", fontsize=13)

import sympy as sp
import math
import sympy as sp

s = sp.symbols('s')
magn = bpmeth.FieldExpansion(b=(0.3, 0.2))

Bx, By, Bs = magn.get_Bfield()

xx = np.linspace(-0.1, 0.1, 50)
yy = np.linspace(-0.1, 0.1, 50)
ss = np.linspace(0, 0.1, 10)

magn_fm = magn.create_fieldmap(xx, yy, ss)
    
fig, ax = plt.subplots()
magn_fm.s_multipoles(3, ax=ax)
magn_fm.s_multipoles(3, ax=ax, method="finite_difference")
plt.legend(fontsize=13)
ax.set_xlabel("s [m]", fontsize=13)
ax.set_ylabel(r"Multipole strength [1/m$^n$]", fontsize=13)
plt.show()
>>>>>>> 3557e1dc55d8087048f90e341f17193427723c69

