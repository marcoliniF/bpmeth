import bpmeth
import numpy as np

quad = bpmeth.FieldExpansion(b=(0, 0.2))
xarr = np.linspace(-0.1, 0.1, 100)
yarr = np.linspace(-0.1, 0.1, 100)
sarr = np.linspace(-1, 1, 100)
quad_fm = quad.create_fieldmap(xarr, yarr, sarr)

