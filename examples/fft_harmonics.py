import bpmeth
import numpy as np
import matplotlib.pyplot as plt

quad = bpmeth.FieldExpansion(b=(0, 0.2))
xarr = np.linspace(-0.1, 0.1, 100)
yarr = np.linspace(-0.1, 0.1, 100)
sarr = np.linspace(-1, 1, 100)
quad_fm = quad.create_fieldmap(xarr, yarr, sarr)

fig, ax = plt.subplots()
quad_fm.s_multipoles(3, ax=ax)

