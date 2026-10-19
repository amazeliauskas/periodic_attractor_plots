import numpy as np

# ==========================================
# Background Evolution (Metric)
# ==========================================

def scale_factor(t, A, omega):
    """Calculates the scale factor a(t) = 1 + A*sin(omega*t)."""
    return 1 + A * np.sin(omega * t)

def expansion_rate(t, A, omega):
    """
    Calculates the expansion rate H(t) = dot{a}/a.
    """
    return A * omega * np.cos(omega * t) / scale_factor(t, A, omega)

# ==========================================
# Thermodynamics & Transport
# ==========================================

def Eeq(T):
    """Equilibrium energy density (massless Boltzmann particles)."""
    return np.pi**2 / 30 * T**4

def findT(E):
    """Inverts the energy density relation to find Temperature."""
    # E = pi^2/30 * T^4  =>  T = (30*E / pi^2)^(1/4)
    return (E / (np.pi**2 / 30))**0.25

def pressure(e):
    """Equation of State: P = e/3."""
    return e / 3

def shear_viscosity(e, eta_s):
    """
    Calculates shear viscosity eta.
    Assumes eta/s = constant (eta_s).
    """
    T = findT(e)
    # Relation: eta = eta_s * (s) = eta_s * (4/3 * e / T)
    return eta_s * (4/3) * (e / T)

def lambda1(e, etas):
        return 5/7*shear_viscosity(e, etas)*relaxation_time(e, etas)

def relaxation_time(e, eta_s):
    """
    Calculates relaxation time tau_pi.
    Common parametrization: tau_pi = 5 * eta / (e + P) = 5 * eta_s / T
    """
    T = findT(e)
    return 5 * eta_s / T

# ==========================================
# Dynamical Equations (ODE Systems)
# ==========================================

def dydt_MIS(t, y, A, omega, eta_s, method="MIS"):
    """
    System of ODEs for Muller-Israel-Stewart (MIS) theory.
    y[0]: Energy density (e)
    y[1]: Shear stress (pi)
    """
    e = y[0]
    pi = y[1]
    
    H = expansion_rate(t, A, omega)
    eta_val = shear_viscosity(e, eta_s)
    tau_val = relaxation_time(e, eta_s)
    lambda_val = lambda1(e, eta_s) 
    # Energy conservation: de/dt = -H * (e + P + pi)
    dedt = -H * (e + pressure(e) + pi)
    
    # Shear evolution (Israel-Stewart): d_pi/dt = - (pi + 4/3*eta*H) / tau_pi
    # Note: This ignores higher order nonlinear terms often found in full IS theory.
    
    if(method == "lambda"):
        #dpidt = -(pi + (4/3) * eta_val * H - lambda_val * 8/9*H**2) / tau_val -(4/3) * H *pi
        #dpidt = -(pi + (4/3) * eta_val * H - lambda_val * 8/9*H**2) / tau_val +(16/9) * H **2 *eta_val

        dpidt = -(pi + (4/3) * eta_val * H - lambda_val / (2*eta_val**2) *pi**2) / tau_val -(4/3) * H *pi
        #dpidt = -(pi + (4/3) * eta_val * H - lambda_val / (2*eta_val**2) *pi**2) / tau_val +(16/9) * H **2 *eta_val
    
        #dpidt = -(pi + (4/3) * eta_val * H + lambda_val *(4/3) * eta_val * H/ (2*eta_val) *pi) / tau_val -(4/3) * H *pi
        #dpidt = -(pi + (4/3) * eta_val * H + lambda_val *(4/3) * eta_val * H/ (2*eta_val) *pi) / tau_val +(16/9) * H **2 *eta_val
    if(method == "MIS"):
        dpidt = -(pi + (4/3) * eta_val * H + 0*lambda_val / (2*eta_val**2) *pi**2) / tau_val -0*(4/3) * H *pi
    return [dedt, dpidt]

def dydt_NS_heating(t, y, A, omega, eta_s):
    """
    ODE for viscous heating correction in Navier-Stokes.
    Used to compute the temperature evolution T_NS.
    """
    H = expansion_rate(t, A, omega)
    a = scale_factor(t, A, omega)
    
    # Based on user code: returns term proportional to eta * H^2 * a^(1/3)
    return [(4/9) * eta_s * (H**2) * (a**(1/3))]

# ==========================================
# Analytical Solutions (Free Streaming)
# ==========================================

def F_func(b):
    """Helper function for Free Streaming anisotropy."""
    # Use np.where to handle arrays and singularities safely
    b = np.array(b, dtype=float)
    res = np.zeros_like(b)
    
    # Case b > 1
    mask_gt = b**2 > 1
    if np.any(mask_gt):
        val = b[mask_gt]
        res[mask_gt] = 0.5 * (1/val**2 + np.arctan(np.sqrt(val**2 - 1)) / np.sqrt(val**2 - 1))
        
    # Case b < 1
    mask_lt = b**2 < 1
    if np.any(mask_lt):
        val = b[mask_lt]
        res[mask_lt] = 0.5 * (1/val**2 + np.arctanh(np.sqrt(1 - val**2)) / np.sqrt(1 - val**2))
        
    # Case b == 1
    mask_eq = np.isclose(b**2, 1.0)
    res[mask_eq] = 1.0
    
    return res

def Pi_P_free_streaming(b):
    """Analytical solution for normalized shear stress in free streaming."""
    F_val = F_func(b)
    
    # Avoid division by zero at b=1
    numerator = (F_val - 1/b**2)
    denominator = (b**2 - 1) * (F_val/3)
    
    # Handle b -> 1 limit if necessary, otherwise compute directly
    # For arrays, we fill safe values then correct
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator - 1.0
        
    # Fix singularity at b=1 (limit is 0)
    if np.isscalar(b):
        if np.isclose(b, 1.0): return 0.0
    else:
        result[np.isclose(b, 1.0)] = 0.0
        
    return result
