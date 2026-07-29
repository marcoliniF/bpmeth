import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import bpmeth
import sympy as sp
import pandas as pd
from scipy.interpolate import PchipInterpolator
a_curved = [0,]
b_curved = [1, 0.02]
h= b_curved[0]# considering normalization B/Brho=1/rho=h            =1. if checking for a drift
print("curvature h[m^-1]: ", h, "radius of curvature [m]: ", 1/h)
#I want to track protons of given kinetic energy
Ek=1e8 #eV
p=np.sqrt(Ek**2+2*Ek*xt.PROTON_MASS_EV)#momentum eV/c
Eref = Ek + xt.PROTON_MASS_EV
print("E [GeV]: ", Eref/1e9)
gamma0 = Eref/xt.PROTON_MASS_EV
beta0 = np.sqrt(1-1/gamma0**2)
print("beta0: ", beta0)
#hx_max=5e-4 #maximum modulus of product of h and x between for initial conditions, to avoid the singularity in the vector potential
xinit=np.linspace(-1e-3, 1e-3, 5)
print("xinit=",xinit)
p0 =xt.Particles(x=xinit, y=0, s=0, py=np.linspace(0.005,-0.005,5), px=np.linspace(0.005,-0.005,5), ptau=0, beta0=beta0, energy0=Eref, mass0=xt.PROTON_MASS_EV)
p_1 = p0.copy()
nphi=5 #default is 5
print("nphi=",nphi)
A_curved= bpmeth.GeneralVectorPotential(h=h, b=b_curved, a=a_curved, nphi=nphi)
#bending_angle=0.3#rad, fix for all the tests
fixedlength=1#bending_angle/h #curved path length
Lstr=0.8#np.sin(bending_angle)/h
H_curved = bpmeth.Hamiltonian(length=fixedlength, h=h, vectp=A_curved)

sol_curved = H_curved.track(p_1, return_sol=True, ivp_opt= {"rtol":1e-10, "atol":1e-12})

# for ss in sol_curved:
#     plt.figure()
#     plt.plot(ss.t, ss.y[0])
# plt.xlabel('s [m]')
# plt.ylabel('x [m]')
# plt.title("Trajectory seen from curved frame")
# x=np.linspace(0,fixedlength,50)
# y= np.zeros_like(x)
# plt.figure()
# plt.plot(x,y, label='arc of circumference of radius $h^{-1}$')
# plt.legend()

# plt.show(block=False)
# plt.pause(0.001)
expansion_curved = bpmeth.FieldExpansion(a=a_curved, b=b_curved, h=h, nphi=5)#, nphi=nphi)
phi_curved = expansion_curved.get_phi()
#print("phi curved frame: ", phi_curved)
#transform the scalar potential to curved frame using coordinate transformations
#xcurved =sqrt((x+rho)**2+s**2)-rho       scurved=rho*arctan(s/(x+rho))   with rho=1/h
x, y, s = sp.symbols('x y s', real=True)
xtmp, stmp = sp.symbols("xtmp stmp", real=True)
# rename the curved coordinates temporarily
phi_tmp = phi_curved.xreplace({
    x: xtmp,
    s: stmp
})
# substitute curved -> straight
phi_straight = phi_tmp.xreplace({
    xtmp: sp.sqrt((x + 1/h)**2 + s**2) - 1/h,
    stmp: 1/h * sp.atan(s/(x + 1/h))
})
#print("\nphi straight frame: ", phi_straight)

Bxstraight_ana = -phi_straight.diff(x)
Bystraight_ana = -phi_straight.diff(y)
Bsstraight_ana = -phi_straight.diff(s)
#print("\nBx straight frame: ", Bxstraight_ana)
def compute_rms_for_order(maxordx, sol_strfromphi, sol_curved, Lstr, make_particle_plots=False):


    dx_exit_list = []
    rms_x_list = []

    for i, (sb, sa) in enumerate(zip(sol_strfromphi, sol_curved)):
        sbar_str = sb.t
        xbar_str = sb.y[0]

        s_curv = sa.t
        x_curv = sa.y[0]

        sbar_from_curv = (rho + x_curv) * np.sin(s_curv / rho)
        xbar_from_curv = -rho + (rho + x_curv) * np.cos(s_curv / rho)

        sbar_min = max(sbar_str.min(), sbar_from_curv.min())
        sbar_max = min(sbar_str.max(), sbar_from_curv.max())

        if sbar_max <= sbar_min:
            print(f"particle {i}: no overlap in sbar range, skipping")
            continue

        sbar_common = np.linspace(sbar_min, sbar_max, 500)

        xbar_str_common = PchipInterpolator(sbar_str, xbar_str)(sbar_common)

        idx_curv = np.argsort(sbar_from_curv)
        sbar_sorted = sbar_from_curv[idx_curv]
        xbar_sorted = xbar_from_curv[idx_curv]

        xbar_curv_common = PchipInterpolator(sbar_sorted, xbar_sorted)(sbar_common)

        dx_all = xbar_str_common - xbar_curv_common
        rms_x = np.sqrt(np.mean(dx_all**2))
        rms_x_list.append(rms_x)

        xbar_exit_straight = np.interp(sbar_fin, sbar_str, xbar_str)
        xbar_exit_from_curv = np.interp(sbar_fin, sbar_sorted, xbar_sorted)

        dx_exit = xbar_exit_straight - xbar_exit_from_curv
        dx_exit_list.append(dx_exit)

        print(
            f"[order {maxordx}] particle {i}: "
            f"dx_exit = {dx_exit:.3e} m, RMS_x = {rms_x:.3e} m"
        )

    mean_dx_exit = np.mean(dx_exit_list) if dx_exit_list else np.nan
    mean_rms_x = np.mean(rms_x_list) if rms_x_list else np.nan

    if make_particle_plots:
        plt.figure()
        plt.plot(range(len(dx_exit_list)), dx_exit_list, "o")
        plt.xlabel("particle index")
        plt.ylabel(r"$\Delta x_{\mathrm{exit}}$ [m]")
        plt.title(
            "Difference at exit using "
            + r"$\phi$ transformation of order "
            + str(maxordx)
        )
        plt.grid(True)

        plt.figure()
        plt.plot(range(len(rms_x_list)), rms_x_list, "o")
        plt.xlabel("particle index")
        plt.ylabel(r"RMS$_x$ over magnet [m]")
        plt.title(
            "RMS difference in x along magnet using "
            + r"$\phi$ transformation of order "
            + str(maxordx)
        )
        plt.grid(True)
        plt.show(block=False)
    plt.pause(0.001)

    return {
        "maxordx": maxordx,
        "mean_dx_exit": mean_dx_exit,
        "mean_rms_x": mean_rms_x,
        "dx_exit_list": dx_exit_list,
        "rms_x_list": rms_x_list,
    }
def compute_rms_curved_for_order(maxordx, sol_strfromphi, sol_curved, make_particle_plots=False):
    rms_x_list = []

    for i, (sb, sa) in enumerate(zip(sol_strfromphi, sol_curved)):
        # --- straight-frame tracking ---
        sbar_str = sb.t
        xbar_str = sb.y[0]

        # --- curved-frame tracking ---
        s_curv = sa.t
        x_curv = sa.y[0]

        # --- transform straight-frame trajectory into curved frame ---
        # from:
        #   sbar = (rho + x) * sin(s/rho)
        #   xbar = -rho + (rho + x) * cos(s/rho)
        #
        # one gets:
        #   rho + x = sqrt((rho + xbar)^2 + sbar^2)
        #   s = rho * atan2(sbar, rho + xbar)

        x_curv_from_str = np.sqrt((rho + xbar_str)**2 + sbar_str**2) - rho
        s_curv_from_str = rho * np.arctan2(sbar_str, rho + xbar_str)

        # --- common curved-frame s range for RMS ---
        s_min = max(s_curv.min(), s_curv_from_str.min())
        s_max = min(s_curv.max(), s_curv_from_str.max())

        if s_max <= s_min:
            print(f"particle {i}: no overlap in curved-frame s range, skipping")
            continue

        s_common = np.linspace(s_min, s_max, 500)

        # interpolate true curved-frame tracking
        idx_curv = np.argsort(s_curv)
        s_curv_sorted = s_curv[idx_curv]
        x_curv_sorted = x_curv[idx_curv]

        x_curv_common = PchipInterpolator(s_curv_sorted, x_curv_sorted)(s_common)

        # interpolate transformed straight->curved tracking
        idx_str = np.argsort(s_curv_from_str)
        s_from_str_sorted = s_curv_from_str[idx_str]
        x_from_str_sorted = x_curv_from_str[idx_str]

        x_from_str_common = PchipInterpolator(s_from_str_sorted, x_from_str_sorted)(s_common)

        # --- RMS difference in curved frame ---
        dx_all = x_from_str_common - x_curv_common
        rms_x = np.sqrt(np.mean(dx_all**2))
        rms_x_list.append(rms_x)

        print(f"[order {maxordx}] particle {i}: RMS_x_curved = {rms_x:.3e} m")

    mean_rms_x = np.mean(rms_x_list) if rms_x_list else np.nan

    if make_particle_plots:
        plt.figure()
        plt.plot(range(len(rms_x_list)), rms_x_list, "o")
        plt.xlabel("particle index")
        plt.ylabel(r"RMS$_x$ over magnet [m]")
        plt.title(
            "RMS difference in curved-frame x along magnet using "
            + r"$\phi$ transformation of order "
            + str(maxordx)
        )
        plt.grid(True)
        plt.show(block=False)

    plt.pause(0.001)

    return {
        "maxordx": maxordx,
        "mean_rms_x": mean_rms_x,
        "rms_x_list": rms_x_list,
    }
orders = [2, 3, 4, 5, 6, 7, 8, 9,10]
results = []
results_fromcurved = []
for ordx in orders:
    maxordx = ordx

    # --------------------------------------------------
    # here the code that depends on maxordx and that
    # regenerates sol_strfromphi and sol_curved
    # --------------------------------------------------
    x, y, s = sp.symbols('x y s', real=True)
    ns=101
    phi_straight_exp = sp.expand(phi_straight)
    phi1 = phi_straight_exp.as_coefficients_dict(y)[y]
    #print("phi1=",phi1) 
    ss = np.linspace(0, 1, ns)
    if phi1==0:
        # safety check for when all b_n(s) are zero, one can just create zero arrays with the right shape
    
        b_vals = np.zeros((maxordx+1, ss.size))
    else:
        phi1_series = phi1.series(x, 0, maxordx+1).removeO()
        dct_b = phi1_series.as_coefficients_dict(x)
        # ... then extract a_n(s) as in the previous case
        items_b = sorted(dct_b.items(), key=lambda kv: sp.degree(kv[0], x))

    b_vals = np.zeros((maxordx+1, ss.size))   # include b_0
    if phi1 != 0:
        # 2. Nontrivial phi0: do series in x and extract b_n(s)
        phi1_series = phi1.series(x, 0, maxordx+1).removeO()
        dct_b = phi1_series.as_coefficients_dict(x)

        items_b = sorted(dct_b.items(), key=lambda kv: sp.degree(kv[0], x))
        #plt.figure()
        for k, (xx, vv) in enumerate(items_b):
            if k == 0:
                b_k = -vv                    # b_1(s)
                label = "b_1(s)"
            else:
                b_k = -vv * sp.factorial(k)  # b_k(s)
                label = f"b_{k+1}(s)"

            b_fun = sp.lambdify(s, b_k, 'numpy')
            b_vals[k, :] = b_fun(ss)

            #plt.plot(ss, b_vals[k, :], label=label)
    else:
        # 3. phi1 == 0: everything is zero; optionally still plot flat lines
        #plt.figure()
        for k in range(maxordx+1):
            label = f"b_{k+1}(s)"
            # a_vals is already zero
            
            #plt.plot(ss, b_vals[k, :], label=label)

    # plt.legend()
    # plt.title("b(s) in straight frame from transformation of $\phi$")
    # plt.xlabel("s")
    # plt.show(block=False)
    # plt.pause(0.001)
    ############################################
    phi0 = phi_straight.as_coefficients_dict(y)[1]   # phi_0(x, s)
    #print("phi0=",phi0)        # you'll see 0 in this case

    if phi0 == 0:
        # safety check for when all a_n(s) are zero, one can just create zero arrays with the right shape
        #ss = np.linspace(0, 1, ns)
        a_vals = np.zeros((maxordx+1, ss.size))
    else:
        phi0_series = phi0.series(x, 0, maxordx+1).removeO()
        dct_a = phi0_series.as_coefficients_dict(x)
        # ... then extract a_n(s) as in the previous case
        items_a = sorted(dct_a.items(), key=lambda kv: sp.degree(kv[0], x))

    a_vals = np.zeros((maxordx+1, ss.size))   # include a_0
    if phi0 != 0:
        # 2. Nontrivial phi0: do series in x and extract a_n(s)
        phi0_series = phi0.series(x, 0, maxordx+1).removeO()
        dct_a = phi0_series.as_coefficients_dict(x)

        items_a = sorted(dct_a.items(), key=lambda kv: sp.degree(kv[0], x))
        #plt.figure()
        for k, (xx, vv) in enumerate(items_a):
            if k == 0:
                a_k = -vv                    # a_0(s)
                label = "a_1(s)"
            else:
                a_k = -vv * sp.factorial(k)  # a_k(s)
                label = f"a_{k+1}(s)"

            a_fun = sp.lambdify(s, a_k, 'numpy')
            a_vals[k, :] = a_fun(ss)
            

            #plt.plot(ss, a_vals[k, :], label=label)
    else:
        # 3. phi0 == 0: everything is zero; optionally still plot flat lines
        #plt.figure()
        for k in range(maxordx+1):
            label = f"a_{k+1}(s)"
            # a_vals is already zero
            
            #plt.plot(ss, a_vals[k, :], label=label)

    # plt.legend()
    # plt.title("a(s) in straight frame from transformation of $\phi$")
    # plt.xlabel("s")
    
    # plt.show(block=False)
    # plt.pause(0.001)

    b_fromphifit = []# I need symbolic expressions of the coefficients as input
    a_fromphifit = []
    s = sp.symbols('s')
    sstraightarray = ss
    degree = 10

    for arr, dst in [(b_vals.T, b_fromphifit), (a_vals.T, a_fromphifit)]:
        for j in range(arr.shape[1]):
            cheb = np.polynomial.Chebyshev.fit(sstraightarray, arr[:, j], deg=degree, domain=[0, 1])
            poly = cheb.convert(kind=np.polynomial.Polynomial)
            coeffs = poly.coef
            poly_expr = sum(sp.Float(coeffs[i]) * s**i for i in range(len(coeffs)))
            dst.append(sp.expand(poly_expr))   # store SymPy expr, not str

    # plt.figure()
    # plt.plot(sstraightarray, b_vals.T[:, 0], label=r"symbolic original b1 from $\phi$")
    # # use b_fromphifit (not b_nfit), index 0 = b_1(s)
    # b1ofsfromphilambdified = sp.lambdify(s, b_fromphifit[0], modules='numpy')
    # plt.plot(sstraightarray, b1ofsfromphilambdified(sstraightarray)*np.ones_like(sstraightarray), '--', label='recovered b1')
    # plt.legend()
    # plt.show(block=False)
    # plt.pause(0.001)
    
    Astrfromphi=bpmeth.GeneralVectorPotential(h=0, b=b_fromphifit, a=a_fromphifit, nphi=maxordx+4)#some margin in nphi
    Hstrfromphi = bpmeth.Hamiltonian(length=Lstr, h=0, vectp=Astrfromphi)
    p_4 = p0.copy()
    sol_strfromphi = Hstrfromphi.track(p_4, return_sol=True, ivp_opt= {"rtol":1e-10, "atol":1e-12})
    # plt.figure()
    # for ss in sol_strfromphi:
    #     plt.plot(ss.t, ss.y[0])
    # plt.xlabel('s [m]')
    # plt.ylabel('x [m]')
    # plt.title("More accurate tracking straight frame(from phi)")
    # x=np.linspace(0,Lstr,50)
    # y= -1/h +np.sqrt(h**-2 -x**2)
    # plt.plot(x,y, color='red', label='arc of circumference of radius $h^{-1}$')
    # #plt.xlim((0., .98))#because H curved has length 1
    # plt.legend()
    # plt.show(block=False)
    # plt.pause(0.001)

    # plt.figure()
    #visualise from the pov of the curved frame
    # for ss in sol_strfromphi:
    #     xc=np.sqrt((ss.y[0]+1/h)**2+ss.t**2)-1/h
    #     sc=np.arctan(ss.t/(ss.y[0]+1/h))/h
    #     plt.plot(sc,xc)
    # plt.xlabel('s [m]')
    # plt.ylabel('x [m]')
    # plt.title("More accurate seen from curved frame")
    # x=np.linspace(0,1,50)
    # y= np.zeros(50)
    # plt.plot(x,y, color='red', label='arc of circumference of radius h^-1')
    # plt.figure()
    # for ss in sol_curved:
        
    #     plt.plot(ss.t, ss.y[0],'--',label='tracking in curved frame')
    # plt.legend()
    # #plt.xlim((0., .98))#because H curved has length 1
    # plt.show(block=False)
    # plt.pause(0.001)
    
    sbar_fin = Lstr - 0.05
    rho = 1.0 / h
    out = compute_rms_for_order(
        maxordx=maxordx,
        sol_strfromphi=sol_strfromphi,
        sol_curved=sol_curved,
        Lstr=Lstr,
        make_particle_plots=False
    )
    results.append(out)
    out_fromcurved= compute_rms_curved_for_order(
        maxordx=maxordx,
        sol_strfromphi=sol_strfromphi,
        sol_curved=sol_curved,
        make_particle_plots=False
    )
    results_fromcurved.append(out_fromcurved)

df_conv = pd.DataFrame({"maxordx": [r["maxordx"] for r in results], "mean_dx_exit": [r["mean_dx_exit"] for r in results],
                         "mean_rms_x": [r["mean_rms_x"] for r in results],
                        "mean_rms_x_fromcurved": [r["mean_rms_x"] for r in results_fromcurved]})


mean_rms_x=df_conv["mean_rms_x"].values #mean_rms_x_h10= [1.30268505e-10 1.30248410e-10 1.30247907e-10 1.30247896e-10 1.30247895e-10 1.30247895e-10]
print("mean_rms_x=",mean_rms_x)
plt.figure()
plt.plot(df_conv["maxordx"], mean_rms_x, "o", label=r"$\Delta x_{\mathrm{exit}}$")
plt.title('Convergence for h='+str(h)+'/m seen from straight frame')
plt.xlabel('# terms in the expansion')
plt.grid(True)
plt.ylabel('mean RMS $x_{straight}$ [m]')
plt.yscale('log')
plt.show(block=False)
plt.pause(0.001)

mean_rms_x_curved=df_conv["mean_rms_x_fromcurved"].values #mean_rms_x_h10= [1.30268505e-10 1.30248410e-10 1.30247907e-10 1.30247896e-10 1.30247895e-10 1.30247895e-10]
print("mean_rms_x curved=",mean_rms_x_curved)
plt.figure()
plt.plot(df_conv["maxordx"], mean_rms_x_curved, "o")
plt.title('Convergence for h='+str(h)+'/m seen from curved frame')
plt.xlabel('# terms in the expansion')
plt.grid(True)
plt.ylabel('mean RMS x [m]')
plt.yscale('log')
plt.show(block=False)
print("done visalisation of convergence")
plt.pause(0.001)
print("now let's fit")
############################################## FIT #####################################################
from scipy.optimize import curve_fit
# Prepare data

orders=[2, 3, 4, 5, 6, 7, 8, 9,10]
orders_arr = np.array(orders)
mean_rms_x_curved= [5.09329580e-07, 7.28402546e-08, 8.84989448e-09, 6.49769258e-10, 8.41810736e-11 ,5.33244687e-11 ,3.07666867e-11 ,2.86895711e-11, 2.86432182e-11]
rmsx_arr = np.array(mean_rms_x_curved)  # Use the curved frame RMS values for fitting
# Define multiple mathematical models
def exp_decay(x, a, b):
    """Exponential decay: a * exp(-b * x)"""
    return a * np.exp(-b * x)

def power_law(x, a, b):
    """Power law: a * x^(-b)"""
    return a * np.power(x, -b)

def stretched_exp(x, a, b, c):
    """Stretched exponential: a * exp(-(b*x)^c)"""
    return a * np.exp(-(b * x)**c)

def inverse_log(x, a, b):
    """Inverse logarithmic: a / (log(x))^b"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return a / np.power(np.log(x), b)

def poly_decay(x, a, b, c):
    """Polynomial decay: a / (1 + b*x^c)"""
    return a / (1 + b * np.power(x, c))

def exp_with_floor(x, a, b, floor):
    """Exponential with floor: a * exp(-b * x) + floor"""
    return a * np.exp(-b * x) + floor

print("=" * 70)
print("TESTING MULTIPLE MODELS FOR RMS_X CURVE")
print("=" * 70)

results = {}

# 1. Exponential decay
print("\n1. EXPONENTIAL DECAY: y = a * exp(-b * x)")
try:
    popt, pcov = curve_fit(exp_decay, orders_arr, rmsx_arr, p0=[1e-4, 0.1], maxfev=5000)
    y_fit = exp_decay(orders_arr, *popt)
    ss_res = np.sum((rmsx_arr - y_fit)**2) 
    ss_tot = np.sum((rmsx_arr - np.mean(rmsx_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr - y_fit)**2))
    results['Exponential'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}")
    print(f"   R² = {r2:.6f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}")

# 2. Power law
print("\n2. POWER LAW: y = a * x^(-b)")
try:
    popt, pcov = curve_fit(power_law, orders_arr, rmsx_arr, p0=[1e-3, 3], maxfev=5000)
    y_fit = power_law(orders_arr, *popt)
    ss_res = np.sum((rmsx_arr - y_fit)**2)
    ss_tot = np.sum((rmsx_arr - np.mean(rmsx_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr - y_fit)**2))
    results['Power Law'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}")
    print(f"   R² = {r2:.6f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}")

# 3. Stretched exponential
print("\n3. STRETCHED EXPONENTIAL: y = a * exp(-(b*x)^c)")
try:
    popt, pcov = curve_fit(stretched_exp, orders_arr, rmsx_arr, p0=[1e-4, 0.2, 0.7], maxfev=5000)
    y_fit = stretched_exp(orders_arr, *popt)
    ss_res = np.sum((rmsx_arr - y_fit)**2)
    ss_tot = np.sum((rmsx_arr - np.mean(rmsx_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr - y_fit)**2))
    results['Stretched Exp'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}, c = {popt[2]:.6f}")
    print(f"   R² = {r2:.8f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}")

# 4. Inverse logarithmic
print("\n4. INVERSE LOGARITHMIC: y = a / (log(x))^b")
try:
    popt, pcov = curve_fit(inverse_log, orders_arr, rmsx_arr, p0=[1e-9, 1], maxfev=5000)
    y_fit = inverse_log(orders_arr, *popt)
    ss_res = np.sum((rmsx_arr - y_fit)**2)
    ss_tot = np.sum((rmsx_arr - np.mean(rmsx_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr - y_fit)**2))
    results['Inverse Log'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}")
    print(f"   R² = {r2:.8f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}")

# 5. Polynomial decay
print("\n5. POLYNOMIAL DECAY: y = a / (1 + b*x^c)")
try:
    popt, pcov = curve_fit(poly_decay, orders_arr, rmsx_arr, p0=[1e-5, 0.01, 2], maxfev=5000)
    y_fit = poly_decay(orders_arr, *popt)
    ss_res = np.sum((rmsx_arr - y_fit)**2)
    ss_tot = np.sum((rmsx_arr - np.mean(rmsx_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr - y_fit)**2))
    results['Poly Decay'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}, c = {popt[2]:.6f}")
    print(f"   R² = {r2:.8f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}") 

# 6. Exponential with floor
print("\n6. EXPONENTIAL WITH FLOOR: y = a * exp(-b * x) + floor")
try:
    popt, pcov = curve_fit(exp_with_floor, orders_arr, rmsx_arr, p0=[1e-4, 1, 1e-10], maxfev=5000)
    y_fit = exp_with_floor(orders_arr, *popt)
    ss_res = np.sum((rmsx_arr - y_fit)**2)
    ss_tot = np.sum((rmsx_arr - np.mean(rmsx_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr - y_fit)**2))
    results['Exp w/ Floor'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}, floor = {popt[2]:.6e}")
    print(f"   R² = {r2:.8f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}")

# Ranking
print("\n" + "=" * 70)
print("MODEL RANKING BY R²")
print("=" * 70)
sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
for i, (model_name, data) in enumerate(sorted_results, 1):
    print(f"{i}. {model_name:20s} | R² = {data['r2']:.8f} | RMSE = {data['rmse']:.6e}")

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (model_name, data) in enumerate(sorted_results[:6]):
    ax = axes[idx]
    ax.semilogy(orders_arr, rmsx_arr, 'o', markersize=8, label='Data', color='blue', zorder=3)
    ax.semilogy(orders_arr, data['y_fit'], '-', linewidth=2.5, 
                label=f"{model_name} ($R^2={data['r2']:.8f}$)", color='red', zorder=2)
    ax.set_xlabel('Maximum order', fontsize=11)
    ax.set_ylabel(r'RMS$_x$ over magnet [m]', fontsize=11)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
best_model = sorted_results[0]
print(f"\nBest fit: {best_model[0]}")
print(f"R² = {best_model[1]['r2']:.8f}")
print("\nThe plateau in the RMS curve (orders 7-12) indicates:")
print("- Exponential decay dominates early (orders 2-6)")
print("- Eventually hits a numerical precision floor")
print("- Need a model that accounts for both regimes")
#now let's fit the best model, the stretched exponential using only orders from 2 to 6
# 3. Stretched exponential
print("\n3. STRETCHED EXPONENTIAL: y = a * exp(-(b*x)^c)")
try:
    popt, pcov = curve_fit(stretched_exp, orders_arr[:5], rmsx_arr[:5], p0=[1e-4, 0.2, 0.7], maxfev=5000)
    y_fit = stretched_exp(orders_arr[:5], *popt)
    ss_res = np.sum((rmsx_arr[:5] - y_fit)**2)
    ss_tot = np.sum((rmsx_arr[:5] - np.mean(rmsx_arr[:5]))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((rmsx_arr[:5] - y_fit)**2))
    results['Stretched Exp'] = {'r2': r2, 'rmse': rmse, 'params': popt, 'y_fit': y_fit}
    print(f"   a = {popt[0]:.6e}, b = {popt[1]:.6f}, c = {popt[2]:.6f}")
    print(f"   R² = {r2:.8f}, RMSE = {rmse:.6e}")
except Exception as e:
    print(f"   Failed: {e}")
#plot the fit
plt.figure(figsize=(8, 5))
plt.semilogy(orders_arr[:5], rmsx_arr[:5], 'o', markersize=8)
plt.semilogy(orders_arr[:5], y_fit, '-', linewidth=2.5, label=f"Stretched Exp Fit ($R^2={r2:.8f}$)", color='red')
plt.xlabel('Maximum order', fontsize=11)
plt.ylabel(r'RMS$_x$ over magnet [m]', fontsize=11)
plt.title('Stretched Exponential Fit for Orders 2-6', fontsize=12, fontweight='bold')
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()