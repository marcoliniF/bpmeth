import matplotlib.pyplot as plt
import bpmeth
import sympy as sp

xmin, xmax, xstep = -0.1, 0.1, 0.01
ymin, ymax, ystep = -0.1, 0.1, 0.01
scale=25
bmin, bmax = -5, 5

##############################
# Normal and skew multipoles #
##############################

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
b1val, b2val, b3val = 1, 25, 500
a1val, a2val, a3val = 1, 25, 500

dipole = bpmeth.FieldExpansion(b=(b1val, ))
dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,0].set_title("Dipole", fontsize=14)

quadrupole = bpmeth.FieldExpansion(b=(0, b2val))
quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0,1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,1].set_title("Quadrupole", fontsize=14)

sextupole = bpmeth.FieldExpansion(b=(0, 0, b3val))
sextupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0,2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,2].set_title("Sextupole", fontsize=14)

sk_dipole = bpmeth.FieldExpansion(a=(a1val, ))
sk_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,0].set_title("Skew dipole", fontsize=14)

sk_quadrupole = bpmeth.FieldExpansion(a=(0, a2val))
sk_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,1].set_title("Skew quadrupole", fontsize=14)

sk_sextupole = bpmeth.FieldExpansion(a=(0, 0, a3val))
sk_sextupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,2].set_title("Skew sextupole", fontsize=14)

plt.tight_layout()
#plt.savefig("Field_derivatives_body.png", dpi=300)
plt.close()

#######################
# Fringe field region #
#######################

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
s = sp.symbols('s')

gap = 0.076
fint = 0.42

bfun = 1/2 * (1 + sp.tanh(s/(fint*2*gap)))

b1fun = bfun * b1val
b2fun = bfun * b2val
b3fun = bfun * b3val
a1fun = bfun * a1val
a2fun = bfun * a2val
a3fun = bfun * a3val

s_dipole = bpmeth.FieldExpansion(b=(b1fun, ))
s_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,0].set_title("Dipole fringe", fontsize=14)

s_quadrupole = bpmeth.FieldExpansion(b=(0, b2fun))
s_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0,1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,1].set_title("Quadrupole fringe", fontsize=14)

s_sextupole = bpmeth.FieldExpansion(b=(0, 0, b3fun))
s_sextupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0,2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,2].set_title("Sextupole fringe", fontsize=14)

s_sk_dipole = bpmeth.FieldExpansion(a=(a1fun, ))
s_sk_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,0].set_title("Skew dipole fringe", fontsize=14)

s_sk_quadrupole = bpmeth.FieldExpansion(a=(0, a2fun))
s_sk_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,1].set_title("Skew quadrupole fringe", fontsize=14)

s_sk_sextupole = bpmeth.FieldExpansion(a=(0, 0, a3fun))
s_sk_sextupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,2].set_title("Skew sextupole fringe", fontsize=14)

plt.tight_layout()
#plt.savefig("Field_derivatives_fringe.png", dpi=300)
plt.close()

#################
# Curved magnet #
#################

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

h = 2

dipole = bpmeth.FieldExpansion(b=(b1val, ), h=h)
dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,0].set_title("Curved dipole", fontsize=14)

quadrupole = bpmeth.FieldExpansion(b=(0, b2val), h=h)
quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0,1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,1].set_title("Curved quadrupole", fontsize=14)

sextupole = bpmeth.FieldExpansion(b=(0, 0, b3val), h=h)
sextupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0,2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,2].set_title("Curved sextupole", fontsize=14)

sk_dipole = bpmeth.FieldExpansion(a=(a1val, ), h=h)
sk_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,0].set_title("Curved skew dipole", fontsize=14)

sk_quadrupole = bpmeth.FieldExpansion(a=(0, a2val), h=h)
sk_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,1].set_title("Curved skew quadrupole", fontsize=14)

sk_sextupole = bpmeth.FieldExpansion(a=(0, 0, a3val), h=h)
sk_sextupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1,2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,2].set_title("Curved skew sextupole", fontsize=14)

plt.tight_layout()
#plt.savefig("Field_derivatives_curved.png", dpi=300)
plt.close()


#########################
# Comparison for dipole #
#########################

scale = 25

fig, ax = plt.subplots(2, 2, figsize=(7.5, 6))

dipole = bpmeth.FieldExpansion(b=(b1val, ))
dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,0].set_title("b1", fontsize=12)

s_dipole = bpmeth.FieldExpansion(b=(b1fun, ))
s_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,1].set_title("b1 with fringe", fontsize=12)

h_dipole = bpmeth.FieldExpansion(b=(b1val, ), h=h)
h_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,0].set_title("b1 with curvature", fontsize=12)

sh_dipole = bpmeth.FieldExpansion(b=(b1fun, ), h=h)
sh_dipole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1, 1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,1].set_title("b1 with curvature and fringe", fontsize=12)

plt.tight_layout()
#plt.savefig("Field_derivatives_dipole.png", dpi=300)
plt.close()

#############################
# Comparison for quadrupole #
#############################

fig, ax = plt.subplots(2, 2, figsize=(7.5, 6))

quadrupole = bpmeth.FieldExpansion(b=(0, b2val))
quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,0].set_title("b2", fontsize=12)

s_quadrupole = bpmeth.FieldExpansion(b=(0, b2fun))
s_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[0, 1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[0,1].set_title("b2 with fringe", fontsize=12)

h_quadrupole = bpmeth.FieldExpansion(b=(0, b2val), h=h)
h_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1, 0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,0].set_title("b2 with curvature", fontsize=12)

sh_quadrupole = bpmeth.FieldExpansion(b=(0, b2fun), h=h)
sh_quadrupole.plot_crossection(bmin=bmin, bmax=bmax, ax=ax[1, 1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xstep=xstep, ystep=ystep, scale=scale)
ax[1,1].set_title("b2 with curvature and fringe", fontsize=12)

plt.tight_layout()
#plt.savefig("Field_derivatives_quadrupole.png", dpi=300)
plt.close()