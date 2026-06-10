import bpmeth
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

def rt2xy(r,theta):
    return r*np.cos(theta),r*np.sin(theta)

def harmonics(ByiBx,nk=5,rmin=0.1,rmax=1,nr=11,ntheta=1024):
    rr=np.linspace(rmin,rmax,nr)
    out=np.empty((len(rr),nk*2+1),dtype=complex)
    theta=np.arange(ntheta)/ntheta*2*np.pi
    for ir,r in enumerate(rr):
      x,y=rt2xy(r,theta)
      b=ByiBx(x,y)
      d=np.fft.fft(b)/ntheta
      out[ir,0]=d[0]
      ii=np.arange(nk)
      out[ir,1::2]=d[1:nk+1]/r**(ii+1)
      out[ir,2::2]=d[ntheta:ntheta-nk-1:-1]/r**(ii+1)
    dd=out[:,0]
    dkl=np.empty((nk*2+1,nk//2+1),dtype=complex)
    dkl[0]=np.polyfit(rr,dd.real,nk)[::-2]+1j*np.polyfit(rr,dd.imag,nk)[::-2]
    for kk in range(0,nk-1):
        dd=out[:,1+2*kk]
        dkl[1+2*kk,:]=np.polyfit(rr,dd.real,nk)[::-2]+1j*np.polyfit(rr,dd.imag,nk)[::-2]
        dd=out[:,2+2*kk]
        dkl[2+2*kk,:]=np.polyfit(rr,dd.real,nk)[::-2]+1j*np.polyfit(rr,dd.imag,nk)[::-2]
    return dkl

def print_harmonics(dkl):
    nk=(dkl.shape[0]-1)//2
    nl=nk//2+1
    print(f"k  ",end="")
    for ll in range(nl):
        print(f"l={ll}"," "*22,end="")
    print()
    print(f" 0",end="")
    for ll in range(nl):
        print(f" {dkl[0,ll].real:12.7f},{dkl[0,ll].imag:12.7f}",end="")
    print()
    for kk in range(0,nk-1):
        print(f" {kk+1}",end="")
        for ll in range(nl):
            print(f" {dkl[1+2*kk,ll].real:12.7f},{dkl[1+2*kk,ll].imag:12.7f}",end="")
        print()
        print(f"-{kk+1}",end="")
        for ll in range(nl):
            print(f" {dkl[2+2*kk,ll].real:12.7f},{dkl[1+2*kk,ll].imag:12.7f}",end="")
        print()

def calc_coeffs(dkl):
    nk = (dkl.shape[0]-1)//2
    nl = nk//2+1
    bnian = np.zeros(nk, dtype=complex)

    for n in range(nk):
        for l in range(n//2+1):
            k = n - 2*l
            if k == 0:
                bnian[n] += math.factorial(n)*dkl[0, l]
            else:
                bnian[n] += math.factorial(n)*dkl[2*k-1, l] + math.factorial(n)*dkl[2*k, l]

    return bnian
            
        
    