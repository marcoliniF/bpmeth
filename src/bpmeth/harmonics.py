<<<<<<< HEAD
import numpy as np
from math import factorial
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial

def dkongrid(Byibx_rts, rr, nk, s_index):
    """
    Takes a precomputed field array and computes d_k as a function of r 
    Parameters
    ----------
    Byibx_rts : complex array, shape (nr, ntheta, ns) representing (By + i*Bx) already evaluated on the (r, theta, s) grid.
    Theta samples must be uniform in [0, 2*pi).
    rr     : array of r values, shape (nr,)
    nk     : max |k|
    s_index: index along the s-axis to use

    Returns
    -------
    out : complex array, shape (nr, 2*nk+1)  — same as dkharmonics
    """
    ntheta = Byibx_rts.shape[1]

    b = Byibx_rts[:, :, s_index]        # (nr, ntheta) at specified s
    d = np.fft.fft(b, axis=1) / ntheta       # FFT along theta axis

    out = np.empty((len(rr), 2*nk+1), dtype=complex)
    out[:, 0]    = d[:, 0]
    out[:, 1::2] = d[:, 1 : nk+1]
    out[:, 2::2] = d[:, ntheta : ntheta-nk-1 : -1]

    return out
def dklfitraw_nodiv_cheb(dkofrarray, rr, nk, nl):                                                                                        #Cg
    """
    Numerically stable Chebyshev version of dklfitraw_nodiv.
    Fits     f_k(r) ≈ Σ_l dkl[k,l] r^{|k|+2l}
    exactly like dklfitraw_nodiv, but internally uses a
    Chebyshev basis in x = r^2 for improved conditioning.
    Returns
    -------
    dkl : complex ndarray, shape (2*nk+1, nl)
        Coefficients with exactly the same meaning as dklfitraw_nodiv:  f_k(r) ≈ Σ_l dkl[k,l] r^{|k|+2l}
    """
    dkl = np.zeros((2*nk + 1, nl), dtype=complex)
    x = rr**2
    xmin = x.min()
    xmax = x.max()
    s = 2*(x - xmin)/(xmax - xmin) - 1 # scale x -> s in [-1,1]
    # Chebyshev Vandermonde in scaled variable
    T = chebvander(s, nl-1)
    # k ordering:
    # [0, +1, -1, +2, -2, ...]
    for ik in range(2*nk + 1):
        if ik == 0:
            kabs = 0
        else:
            kabs = (ik + 1)//2
        # absorb r^|k| into the matrix itself
        A = (rr[:, None]**kabs) * T
        y = dkofrarray[:, ik]
        # stable least squares in Chebyshev basis
        c_re, _, _, _ = np.linalg.lstsq(A, y.real, rcond=None)
        c_im, _, _, _ = np.linalg.lstsq(A, y.imag, rcond=None)
        c = c_re + 1j*c_im
        # Convert:  Σ c_n T_n(s)    into ordinary polynomial in x=r^2: Σ a_l x^l
        cheb = Chebyshev(c, domain=[xmin, xmax])
        poly = cheb.convert(kind=Polynomial)
        coeffs = poly.coef
        # store coefficients of x^l so total basis is    r^|k| * (r^2)^l = r^{|k|+2l}
        ncopy = min(len(coeffs), nl)
        dkl[ik, :ncopy] = coeffs[:ncopy]
    return dkl
def bnianHAopt(dkl, nl, nk):#from dkl to bn and an
    # Precompute factorials
    fact = np.array([factorial(n) for n in range(nk)], dtype=np.int16)

    # Build k index dict mapping: row index for k = 0,1,-1,2,-2,...
    k_index = {0: 0}
    for i in range(1, nk+1):
        k_index[i]  = 2*i - 1
        k_index[-i] = 2*i #negative k gets even row

    result = np.zeros(nk, dtype=np.complex128)

    for n in range(nk):
        s = 0.0 + 0.0j #initialize sum
        max_l = min(n//2, nl-1)

        for l in range(max_l + 1):#
            k = n - 2*l
            if k == 0:
                s += dkl[k_index[0], l]
            else:
                s += dkl[k_index[k], l] + dkl[k_index[-k], l]

        result[n] = fact[n] * s

    return result
def LHongrid_nodiv_cheb(ByiBxongrid, s_index, nk, rr):
    '''
    Complete function for local harmonic analysis from By+iBx to coefficients a,b 
    '''
    dk=dkongrid(Byibx_rts=ByiBxongrid, rr=rr, nk=nk, s_index=s_index)
    nl=nk//2+1
    dkl=dklfitraw_nodiv_cheb(dkofrarray=dk,rr=rr,nk=nk, nl=nl)
    an=bnianHAopt(dkl, nl,nk).imag
    bn=bnianHAopt(dkl, nl,nk).real
    return an, bn

=======
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
>>>>>>> main
            
        
    