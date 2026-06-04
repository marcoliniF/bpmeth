import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import bpmeth
#I want to track protons of given kinetic energy
Ek=1e8 #eV
p=np.sqrt(Ek**2+2*Ek*xt.PROTON_MASS_EV)#momentum eV/c
print("p [GeV/c]: ", p/1e9)
a = [0,]
b = [1,]
h= b[0]# considering normalization B/Brho=1/rho=h
print("curvature h[m^-1]: ", h, "radius of curvature [m]: ", 1/h)

Eref = Ek + xt.PROTON_MASS_EV
print("E [GeV]: ", Eref/1e9)
gamma0 = Eref/xt.PROTON_MASS_EV
beta0 = np.sqrt(1-1/gamma0**2)
print("beta0: ", beta0)
p0 =xt.Particles(x=np.linspace(-1e-3, 1e-3, 5), y=0, s=0,py=0.001, px=0.001, ptau=0, beta0=beta0, energy0=Eref, mass0=xt.PROTON_MASS_EV)
p_2 = p0.copy()

A_straight= bpmeth.GeneralVectorPotential(h=0, b=b, a=a)
H_straight = bpmeth.Hamiltonian(length=1, h=0, vectp=A_straight)

sol_straight = H_straight.track(p_2, return_sol=True, ivp_opt= {"rtol":1e-10, "atol":1e-12},)

for ss in sol_straight:
    plt.plot(ss.t, ss.y[0])
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.title("Trajectory straight frame")
x=np.linspace(0,1,50)
y= -1/h+np.sqrt(h**-2-x**2)
plt.plot(x,y, color='red', label='arc of circumference of radius h^-1')
plt.legend()
plt.show()
A_curved= bpmeth.GeneralVectorPotential(h=h, b=b, a=a)
H_curved = bpmeth.Hamiltonian(length=1, h=h, vectp=A_curved)
p_1 = p0.copy()
sol_curved = H_curved.track(p_1, return_sol=True, ivp_opt= {"rtol":1e-10, "atol":1e-12})

for ss in sol_curved:
    plt.plot(ss.t, ss.y[0])
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.title("Trajectory curved frame h[m^-1]="+str(h))
x=np.linspace(0,1,50)
plt.plot(x,np.zeros_like(x), color='red', label='arc of circumference of radius h^-1')#circumference is a straight line in the curved frame with the same curvature
plt.legend()
plt.show()