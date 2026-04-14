import bpmeth
import numpy as np
import matplotlib.pyplot as plt

quad = bpmeth.FieldExpansion(b=(0.5, 0.3, 0.1))
xarr = np.linspace(-0.1, 0.1, 100)
yarr = np.linspace(-0.1, 0.1, 100)
sarr = np.linspace(0, 1, 10)
quad_fm = quad.create_fieldmap(xarr, yarr, sarr)

#quad_fm.harmonic_analysis_at_s(0, 0.05, order=5)

fig, ax = plt.subplots()
quad_fm.s_multipoles(3, ax=ax)
#quad_fm.s_multipoles(3, ax=ax, method="fft")
quad_fm.s_multipoles(3, ax=ax, method="finite_difference")
plt.legend(fontsize=13)
ax.set_xlabel("s [m]", fontsize=13)
ax.set_ylabel(r"Multipole strength [1/m$^n$]", fontsize=13)
plt.show()

