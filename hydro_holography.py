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

def expansion_rate_dot(t, A, omega):
    """
    Calculates the expansion rate H(t) = dot{dot{a}}/a.
    """
    return -A * omega*omega * np.sin(omega * t) / scale_factor(t, A, omega)
# ==========================================
# Thermodynamics & Transport
# ==========================================

def Eeq(T):
    return np.pi**4 * T**4

def findT(E):
    """Inverts the energy density relation to find Temperature."""
    return (E / (np.pi**4 ))**0.25

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

def relaxation_time(e, eta_s):
    """
    Calculates relaxation time tau_pi.
    Common parametrization: tau_pi = 5 * eta / (e + P) = 5 * eta_s / T
    """
    T = findT(e)
    return (4-2*np.log(2)) * eta_s / T

# ==========================================
# Dynamical Equations (ODE Systems)
# ==========================================

def dydt_MIS(t, y, A, omega, eta_s, kappaT_eta):
    """
    System of ODEs for Muller-Israel-Stewart (MIS) theory.
    y[0]: Energy density (e)
    y[1]: Shear stress (pi)
    """
    e = y[0]
    pi = y[1]
    
    H = expansion_rate(t, A, omega)
    Hdot = expansion_rate_dot(t, A, omega)
    eta_val = shear_viscosity(e, eta_s)
    tau_val = relaxation_time(e, eta_s)
    
    # Energy conservation: de/dt = -H * (e + P + pi)
    dedt = -H * (e + pressure(e) + pi)
    kappa = kappaT_eta*eta_val/findT(e)
    # Shear evolution (Israel-Stewart): d_pi/dt = - (pi + 4/3*eta*H) / tau_pi
    # Note: This ignores higher order nonlinear terms often found in full IS theory.
    dpidt = -(pi + (4/3) * eta_val * H + 2/3*kappa*Hdot ) / tau_val
    
    return [dedt, dpidt]

def dydt_NS_heating(t, y, A, omega, eta_s):
    """
    ODE for viscous heating correction in Navier-Stokes.
    Used to compute the temperature evolution T_NS.
    """
    H = expansion_rate(t, A, omega)
    a = scale_factor(t, A, omega)
    
    # Based on user code: returns term proportional to eta * H^2 * a^(1/3)
    return (4/9) * eta_s * (H**2) * (a**(1/3))

# ==========================================
# Analytical Solutions (Free Streaming)
# ==========================================
