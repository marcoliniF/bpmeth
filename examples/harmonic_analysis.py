import bpmeth
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import sympy as sp

s = sp.symbols('s')
magn = bpmeth.FieldExpansion(b=(0.3, 0.4+s**2, 0.5, 0.6), a=(0.1,))  #b2''=-0.06

Bx, By, Bs = magn.get_Bfield()

#################
# Exact example #
#################

def ByiBx(x, y, s=0):
    return By(x, y, s) + 1j*Bx(x, y, s)

dkl = bpmeth.harmonics(ByiBx)
coeffs = bpmeth.calc_coeffs(dkl)
for i, coeff in enumerate(coeffs):
    print(f"b{i} = {coeff.real:10.6f}, a{i} = {coeff.imag:10.6f}")

############
# Fieldmap #
############

rmin, rmax, nr = 0.1, 0.5, 51
ntheta = 1024
rr = np.linspace(rmin, rmax, nr)
tt = np.arange(ntheta)/ntheta*2*np.pi
ss = np.linspace(0, 0, 1)
rarr, tarr, sarr = np.meshgrid(rr, tt, ss)
xarr = rarr*np.cos(tarr)
yarr = rarr*np.sin(tarr)

Bxvals, Byvals, Bsvals = Bx(xarr, yarr, sarr), By(xarr, yarr, sarr), Bs(xarr, yarr, sarr)

# Gaussian noise
Bxvals += np.random.randn(*Bxvals.shape)*1e-8
Byvals += np.random.randn(*Byvals.shape)*1e-8
Bsvals += np.random.randn(*Bsvals.shape)*1e-8

magn_fm = bpmeth.Fieldmap(np.array([xarr.flatten(), yarr.flatten(), sarr.flatten(), Bxvals.flatten(), Byvals.flatten(), Bsvals.flatten()]).T)

# Method does not seem very numerically stable
coeffs = magn_fm.harmonic_analysis_at_s(0, rmin=rmin, rmax=rmax, nr=nr, ntheta=ntheta, radius=0.001)
for i, coeff in enumerate(coeffs):
    print(f"b{i} = {coeff.real:10.6f}, a{i} = {coeff.imag:10.6f}")
