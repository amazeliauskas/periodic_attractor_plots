import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Import physics model
import hydro_boltzmann as phys

# ==========================================
# Plotting Configuration
# ==========================================

# scienceplots is optional and only changes the looks: if it is missing or fails to
# load (e.g. incompatible with the installed matplotlib), keep the default style.
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'nature'])
except Exception as err:
    print(f"Warning: 'scienceplots' style unavailable ({err!r}). Using default style.")

prop_cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)    
colors = prop_cycle if prop_cycle is not None else plt.cm.Set2(np.linspace(0, 1, 10)) 
                                                                                

# 7 colors with maximum visual separation
colors = [
    "#E6194B",  # Red (Strong & Primary)
    "#3CB44B",  # Green (True Green, not bluish)
    "#4363D8",  # Blue (Royal Blue)
    "#F58231",  # Orange (Warm & Bright)
    "#911EB4",  # Purple (Deep & Rich)
    "#46F0F0",  # Cyan (High Contrast vs Blue)
    "#F032E6"   # Magenta (High Contrast vs Red/Purple)
]
plt.rcParams.update({                                                           
    'font.size': 11,                                                            
    'xtick.minor.visible': True,                                                
    'ytick.minor.visible': True,                                                
    'axes.linewidth': 0.5,                                                      
    'axes.labelsize': 11,                                                       
    'legend.fontsize': 10,                                                      
    'xtick.labelsize': 11,                                                      
    'ytick.labelsize': 11                                                       
})                                                                              
print(colors)
COLORS = colors
LABELS = ['Navier-Stokes', 'Müller-Israel-Stewart', 'free-streaming', 'linear MIS', 'Yang-Mills', 'RTA', 'Scalar']
STYLE = ['-', '-', '--', ':', '-.',':', ':'] 

# Global Plot Settings
FRAME_WIDTH = 2.5
GOLDEN_RATIO = 0.75

# ==========================================
# Simulation Parameters
# ==========================================

# Constants
ETA_S = 0.56
C_TAU_SCALAR = 6.10517  # scalar phi^4: C_tau_pi/C_eta = T tau_pi/(eta/s), Table 1 of paper
T0 = 1.0
WTAU_LIST = [0.1, 0.5, 2.0, 10] # List of omega*tau_pi values to iterate over

# ==========================================
# Plotting Functions
# ==========================================

def plot_expansion_rate(output_name='plot_rate.pdf', outputpath="./paper_figures/"):
    """Plots the normalized expansion rate for different amplitudes A."""
    # Own style for this small single panel (as in the published figure): scienceplots
    # 'science'+'nature' defaults (7pt text) instead of the global 11pt settings, no minor
    # ticks, and the colour cycle of the original plot.py.
    style = {
        'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.prop_cycle': plt.cycler('color', ['#0085CA', '#008F00', '#FF9500', '#FF2C00']),
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 1, figsize=(172.08/72, 127.08/72))

        t_vals = np.linspace(0, 2*np.pi, 1000)
        amplitudes = [0.1, 0.5, 0.8, 0.9]

        for A in amplitudes:
            norm_rate = np.cos(t_vals) / (1 + A * np.sin(t_vals))
            ax.plot(t_vals, norm_rate, label=f'$A={A}$')

        ax.hlines(0, 0, 2*np.pi, ls='-', color='black', lw=0.25)
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([-3, 3])
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xlabel(r'$\omega t$')
        ax.set_ylabel(r'$\dot{a}/(a A\omega)$')
        ax.legend(loc='upper center', frameon=False)   # upper middle: clear of the curves

        plt.tight_layout()
        plt.savefig(outputpath+output_name, pad_inches=0.1)
        plt.close(fig)
    print(f"Saved {output_name}")


def plot_phase_space_comparison(A, load_external_data=False, outputpath="./paper_figures/"):
    """
    Plots the Phase Space diagram: Shear Stress (pi) vs Navier-Stokes Shear (pi_NS).
    Glued version: panels share y-axis and have zero horizontal space.
    """
    scale = 0.8
    n_plots = len(WTAU_LIST)
    # Added sharey=True and wspace=0 to glue the panels horizontally
    fig, axs = plt.subplots(1, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, scale*FRAME_WIDTH), 
                            sharey=True, gridspec_kw={'wspace': 0})
    
    if n_plots == 1: axs = [axs]

    for i, wtau in enumerate(WTAU_LIST):
        ax = axs[i]
        tauR0 = 5 * ETA_S / T0
        omega = wtau / tauR0
        t_max = 2.5 * (2*np.pi/omega)
        t_eval = np.linspace(0, t_max, 1000)

        yfac = 16/3 * ETA_S/T0 * A * omega
        
        # 1. Navier-Stokes (Reference)
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, ETA_S), [0, t_max], y0_NS, t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
        xNS_ref = - (16/15) * (5*ETA_S/T_NS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac
        
        ax.plot(xNS_ref, xNS_ref, color=COLORS[0], ls=STYLE[0], lw=1, label=LABELS[0])

        # 2. External Data (EKT)
        if load_external_data:
            filename = f"./data/YangMills/small_amplitude/YM_lambda10_taupi5.2_A{A}_wtaupi{wtau}.out"
            if os.path.exists(filename):
                data = np.loadtxt(filename)
                t_ekt, e_ekt, pi_ekt = data[:, 0], data[:, 8], data[:, 17] - (data[:, 8]/3.0)
                T_ekt = phys.findT(e_ekt)
                pi_norm_ekt = pi_ekt / phys.pressure(e_ekt)
                xNS_ekt = - (16/15) * (5 * ETA_S / T_ekt) * A * omega * np.cos(omega * t_ekt) / (1 + A*np.sin(omega*t_ekt))
                ax.plot(xNS_ekt / yfac, pi_norm_ekt / yfac, color='red', label='EKT')

        # 3. Non-linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S), [0, t_max], y0_MIS, t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        
        ax.plot(xNS_ref, pi_MIS_norm, color=COLORS[1], ls=STYLE[1], label=LABELS[1])

        # 4. Free Streaming
        a_val = phys.scale_factor(t_eval, A, omega)
        pi_free_norm = phys.Pi_P_free_streaming(a_val) / yfac
        
        ax.plot(xNS_ref, pi_free_norm, color=COLORS[2], ls=STYLE[2], label=LABELS[2])

        # 4. Linear MIS
       	T_lin_MIS = 4/3-a_val/3
        omegat = t_eval*omega
        tauR = 5 * ETA_S / T0
        phi = np.arctan(wtau)
        # Amplitude dampening factor: 1 / sqrt(1 + (w*tau)^2)
        pi_MIS_lin_norm = - (16/15) * (5 * ETA_S / T0) * A * omega *(np.cos(omegat - phi) / np.sqrt(wtau**2 + 1) - np.exp(-t_eval*omega/wtau)/(wtau**2 + 1) )/yfac
        xNS_lin = - (16/15) * (5*ETA_S/T0) *A*omega*np.cos(omegat)/yfac
        
        ax.plot(xNS_lin, pi_MIS_lin_norm, color=COLORS[3], ls=STYLE[3], label=LABELS[3])
        # Formatting
        xmax_val = 1.1 * np.max(np.abs(xNS_ref))
        ax.set_xlim([-xmax_val, xmax_val])
        ax.set_ylim([-xmax_val, xmax_val])
        ax.set_aspect('equal')
        ax.set_title(r'$\omega \tau_\pi = %g$, $A=%g$' % (wtau, A), fontsize=9)
        ax.set_xlabel(r'$\pi_\mathrm{NS}/(P_{eq}\,\mathcal{N})$')

        if i == 0:
            ax.set_ylabel(r'$\pi/(P_{eq}\,\mathcal{N})$')
            ax.legend(fontsize=7, loc='upper left')

    plt.savefig(outputpath + f'plot_phase_space_A{A}.pdf', bbox_inches='tight')
    print(f"Saved plot_phase_space_A{A}.pdf")


def plot_kinetic_temperature_evolution(A, outputpath="./paper_figures/"):
    """
    Plots Temperature and Shear Stress vs Time.
    Glued version: panels share x-axis (col) AND y-axis (row).
    """
    scale = 0.8

    n_plots = len(WTAU_LIST)
    fig, axs = plt.subplots(2, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, scale*2 * FRAME_WIDTH), 
                            sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    if n_plots == 1: axs = axs.reshape(2, 1)

    for i, wtau in enumerate(WTAU_LIST):
        eta_s = ETA_S
        tau_RTA = 5 * eta_s / T0
        omega_RTA = wtau / tau_RTA
        t_max_default = 2.5 * (2*np.pi/omega_RTA)
        t_eval = np.linspace(0, t_max_default, 1000) # Default t_eval if EKT missing
        xfac = 2*np.pi
        xmax = t_max_default*omega_RTA / xfac
        
        t_max = t_max_default
        # List to hold the data sets to plot. 
        # Only populated if the data is successfully loaded/calculated.
        plot_items = []
        # 3. MIS ODE
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega_RTA, eta_s), [0, t_max], y0_MIS, t_eval=t_eval)
        T_MIS = phys.findT(sol_MIS.y[0])
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0])
        
        yfac_RTA = 16/3*eta_s/T0*A*omega_RTA
        plot_items.append({
            'x': sol_MIS.t * omega_RTA/xfac, 'y_T': T_MIS/T0, 'y_pi': pi_MIS_norm/yfac_RTA, 'idx': 1
        })
        # 1. YM kinetics
        filename = f"./data/YangMills/small_amplitude/YM_lambda10_taupi5.2_A{A}_wtaupi{wtau}.out"
        if os.path.exists(filename):
            data = np.loadtxt(filename)
            t_ekt, e_ekt, pi_ekt = data[:, 0], data[:, 8], data[:, 17] - (data[:, 8]/3.0)
            T_ekt = phys.findT(e_ekt)
            pi_norm_ekt = pi_ekt / phys.pressure(e_ekt)
            tau_EKT = 5.2 * eta_s / T0
            omega_EKT = wtau / tau_EKT
            yfac_EKT = 16/3*eta_s/T0*A*omega_EKT
            
            plot_items.append({
                'x': t_ekt * omega_EKT/xfac, 'y_T': T_ekt/T0, 'y_pi': pi_norm_ekt/yfac_EKT, 'idx': 4
            })
        else:
            print("File %s not found" % filename)
        
        # 2. RTA
        filename_RTA=f"./data/RTA/small_amplitude/RTA_etas0.56_A%g_wtaupi%.1f.txt" % (A, wtau)
        if os.path.exists(filename_RTA):
            data = np.loadtxt(filename_RTA)
            t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS, a_RTA, adot_RTA, wt_RTA = data.T
            pi_norm_RTA = 2/3*(pL_RTA-pT_RTA)/(e_RTA/3) 
            
            plot_items.append({
                'x': t_RTA * omega_RTA/xfac, 'y_T': T_RTA/T0, 'y_pi': pi_norm_RTA/yfac_RTA, 'idx': 5
            })
        else:
            print("File %s not found" % filename_RTA)

        # 3. Scalars
        tau_S = C_TAU_SCALAR * eta_s / T0
        omega_S = wtau / tau_S
        # file name encodes the drive frequency f = omega_S (6 decimals), fixing the run unambiguously
        filename_S = f"./data/scalar_phi4/small_amplitude/scalar_A{A}_omega{omega_S:.6f}_wtaupi{wtau:.1f}_ntheta256.txt"
        if os.path.exists(filename_S):
            data = np.loadtxt(filename_S)
            # s t a adot exprate tstar mdsqr n e pl T mu
            t_S = data[:,1]
            e_S = data[:,8]
            pL_S = data[:,9]
            T_S = data[:,10]
            pi_norm_S = pL_S/(e_S/3) -1

            yfac_S = 16/3*eta_s/T0*A*omega_S
            plot_items.append({
                'x': t_S * omega_S/xfac , 'y_T': T_S, 'y_pi': pi_norm_S/yfac_S, 'idx': 6
            })
        else:
            print("File %s not found" % filename_S)

        # --- Top Row (Temperature) ---
        ax_T = axs[0, i]
        ax_T.hlines(1, 0, xmax, 'k', lw=0.5)
        
        # Loop over the populated items to plot
        for item in plot_items:
            idx = item['idx']
            ax_T.plot(item['x'], item['y_T'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])
        
        ax_T.set_title(r'$\omega \tau_\pi = %g$, $A=%g$' % (wtau, A), fontsize=9)
        ax_T.set_ylim([0.96, 1.09])
        if i == 0: 
            ax_T.set_ylabel(r'$T/T_0$')
            ax_T.legend(fontsize=7)

        # --- Bottom Row (Shear) ---
        ax_Pi = axs[1, i]
        ax_Pi.hlines(0, 0, xmax, 'k', lw=0.5)
        
        # Loop over the populated items to plot
        for item in plot_items:
            idx = item['idx']
            ax_Pi.plot(item['x'], item['y_pi'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

        ax_Pi.set_xlabel(r'$\omega t/(2\pi)$')
        ax_Pi.set_xlim([0, xmax])
        ax_Pi.set_ylim([-1.1, 1.1])
        if i == 0: 
            ax_Pi.set_ylabel(r'$\pi/(P_{eq}\,\mathcal{N})$')

    os.makedirs(outputpath, exist_ok=True)
    plt.savefig(outputpath + f'plot_kinetic_T_A{A}.pdf', bbox_inches='tight')
    print(f"Saved plot_kinetic_T_A{A}.pdf")

def plot_MIS_temperature_evolution(A, outputpath="./paper_figures/"):
    """
    Plots Temperature and Shear Stress vs Time.
    Glued version: panels share x-axis (col) AND y-axis (row).
    """
    scale = 0.8
    n_plots = len(WTAU_LIST)
    fig, axs = plt.subplots(2, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, scale*2 * FRAME_WIDTH), 
                            sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    if n_plots == 1: axs = axs.reshape(2, 1)

    for i, wtau in enumerate(WTAU_LIST):
        tauR0 = 5 * ETA_S / T0
        omega = wtau / tauR0
        t_max = 2.5 * (2*np.pi/omega)
        t_eval = np.linspace(0, t_max, 1000)
        xfac = 2*np.pi
        xmax = t_max*omega / xfac
        yfac = 16/3 * ETA_S/T0 * A * omega

        # 1. NS Heating
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, ETA_S), [0, t_max], y0_NS, t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
        
        # 2. MIS ODE
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S), [0, t_max], y0_MIS, t_eval=t_eval)
        T_MIS = phys.findT(sol_MIS.y[0])
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0])

        # 3. Free streaming
        a_val = phys.scale_factor(t_eval, A, omega)
        T_free = phys.F_func(a_val)**0.25
        pi_free_norm = phys.Pi_P_free_streaming(a_val)
        
        # 4. Linear MIS
       	T_lin_MIS = 4/3-a_val/3
        omegat = t_eval*omega
        tauR = 5 * ETA_S / T0
        phi = np.arctan(wtau)
        # Amplitude dampening factor: 1 / sqrt(1 + (w*tau)^2)
        pi_MIS_lin_norm = - (16/15) * (5 * ETA_S / T0) * A * omega * ( (np.cos(omegat - phi)) / np.sqrt(wtau**2 + 1) - np.exp(-t_eval*omega/wtau)/(wtau**2 + 1) )
        
        # --- Top Row (Temperature) ---
        ax_T = axs[0, i]
        ax_T.hlines(1, 0, xmax, 'k', lw=0.5)
        ax_T.plot(sol_NS.t * omega/xfac, T_NS/T0, color=COLORS[0], ls=STYLE[0], label=LABELS[0])
        ax_T.plot(sol_MIS.t * omega/xfac, T_MIS/T0, color=COLORS[1], ls=STYLE[1], label=LABELS[1])
        ax_T.plot(t_eval * omega/xfac, T_free, color=COLORS[2], ls=STYLE[2], label=LABELS[2])
        ax_T.plot(t_eval * omega/xfac, T_lin_MIS, color=COLORS[3], ls=STYLE[3], label=LABELS[3])
        
        ax_T.set_title(r'$\omega \tau_\pi = %g$, $A=%g$' % (wtau, A), fontsize=9)
        ax_T.set_ylim([0.96, 1.09])
        if i == 0: 
            ax_T.set_ylabel(r'$T/T_0$')
            ax_T.legend(fontsize=7)

        # --- Bottom Row (Shear) ---
        ax_Pi = axs[1, i]
        ax_Pi.hlines(0, 0, xmax, 'k', lw=0.5)
        ax_Pi.plot(sol_NS.t * omega/xfac, - (16/15) * (5*ETA_S/T_NS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac, color=COLORS[0], ls=STYLE[0], label=LABELS[0])
        ax_Pi.plot(sol_MIS.t * omega/xfac, pi_MIS_norm / yfac, color=COLORS[1], ls=STYLE[1], label=LABELS[1])
        ax_Pi.plot(t_eval * omega/xfac, pi_free_norm / yfac, color=COLORS[2], ls=STYLE[2], label=LABELS[2])
        ax_Pi.plot(t_eval * omega/xfac, pi_MIS_lin_norm / yfac, color=COLORS[3], ls=STYLE[3], label=LABELS[3])

        ax_Pi.set_xlabel(r'$\omega t/(2\pi)$')
        ax_Pi.set_xlim([0, xmax])
        ax_Pi.set_ylim([-1.1, 1.1])
        if i == 0:
            ax_Pi.set_ylabel(r'$\pi/(P_{eq}\,\mathcal{N})$')

    plt.savefig(outputpath + f'plot_MIS_T_A{A}.pdf', bbox_inches='tight')
    print(f"Saved plot_MIS_T_A{A}.pdf")
def plot_kinetic_phase_space_comparison(A, outputpath="./paper_figures/"):
    """
    Plots the Phase Space diagram: Shear Stress (pi) vs Navier-Stokes Shear (pi_NS).
    Uses the data loading structure from plot_kinetic_temperature_evolution, 
    allowing dynamically loaded YM and RTA data to be plotted alongside ODE solutions.
    """
    scale = 0.8
    n_plots = len(WTAU_LIST)
    
    # Glued version: panels share y-axis and have zero horizontal space
    fig, axs = plt.subplots(1, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, scale*FRAME_WIDTH), 
                            sharey=True, gridspec_kw={'wspace': 0})
    
    if n_plots == 1: axs = [axs]

    for i, wtau in enumerate(WTAU_LIST):
        ax = axs[i]
        eta_s = ETA_S
        tau_RTA = 5 * eta_s / T0
        omega_RTA = wtau / tau_RTA
        t_max_default = 2.5 * (2*np.pi/omega_RTA)
        t_eval = np.linspace(0, t_max_default, 1000)

        # Scaling factor for the axes
        
        yfac_RTA = 16/3*eta_s/T0*A*omega_RTA
        plot_items = []

        # 0. Navier-Stokes (Reference)
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega_RTA, eta_s), [0, t_max_default], y0_NS, t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega_RTA)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
        xNS_ref = - (16/15) * (5*eta_s/T_NS) * phys.expansion_rate(sol_NS.t, A, omega_RTA) / yfac_RTA
        
        # 1. Non-linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega_RTA, eta_s), [0, t_max_default], y0_MIS, t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac_RTA
        
        plot_items.append({
            'x': xNS_ref, 'y': pi_MIS_norm, 'idx': 1
        })

        # 4. YM kinetics (External Data)
        filename_YM = f"./data/YangMills/small_amplitude/YM_lambda10_taupi5.2_A{A}_wtaupi{wtau}.out"
        if os.path.exists(filename_YM):
            data = np.loadtxt(filename_YM)
            t_ekt, e_ekt, pi_ekt = data[:, 0], data[:, 8], data[:, 17] - (data[:, 8]/3.0)
            tau_EKT = 5.2 * eta_s / T0
            omega_EKT = wtau / tau_EKT
            mask = t_ekt * omega_EKT <= 2.5 * 2*np.pi
            t_ekt, e_ekt, pi_ekt = t_ekt[mask], e_ekt[mask], pi_ekt[mask]
            T_ekt = phys.findT(e_ekt)
            pi_norm_ekt = pi_ekt / phys.pressure(e_ekt)
            yfac_EKT = 16/3*eta_s/T0*A*omega_EKT
            # Calculate corresponding xNS axis for EKT time steps
            xNS_ekt = - (16/15) * (5 * eta_s / T_ekt) * A * omega_EKT * np.cos(omega_EKT * t_ekt) / (1 + A*np.sin(omega_EKT*t_ekt))
            
            plot_items.append({
                'x': xNS_ekt / yfac_EKT, 'y': pi_norm_ekt / yfac_EKT, 'idx': 4
            })
        else:
            print("File %s not found" % filename_YM)

        # 5. RTA (External Data)
        filename_RTA = f"./data/RTA/small_amplitude/RTA_etas0.56_A%g_wtaupi%.1f.txt" % (A, wtau)
        if os.path.exists(filename_RTA):
            data = np.loadtxt(filename_RTA)
            t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS_RTA, a_RTA, adot_RTA, wt_RTA = data.T
            mask = t_RTA * omega_RTA <= 2.5 * 2*np.pi
            t_RTA, e_RTA, pT_RTA, pL_RTA, T_RTA = t_RTA[mask], e_RTA[mask], pT_RTA[mask], pL_RTA[mask], T_RTA[mask]
            pi_norm_RTA = 2/3*(pL_RTA-pT_RTA)/(e_RTA/3)

            # Calculate corresponding xNS axis for RTA time steps
            xNS_RTA = - (16/15) * (5 * eta_s / T_RTA) * A * omega_RTA * np.cos(omega_RTA * t_RTA) / (1 + A*np.sin(omega_RTA*t_RTA))
            
            yfac_RTA = 16/3*eta_s/T0*A*omega_RTA
            plot_items.append({
                'x': xNS_RTA / yfac_RTA, 'y': pi_norm_RTA / yfac_RTA, 'idx': 5
            })
        else:
            print("File %s not found" % filename_RTA)
        # 3. Scalars
        tau_S = C_TAU_SCALAR * eta_s / T0
        omega_S = wtau / tau_S
        # file name encodes the drive frequency f = omega_S (6 decimals), fixing the run unambiguously
        filename_S = f"./data/scalar_phi4/small_amplitude/scalar_A{A}_omega{omega_S:.6f}_wtaupi{wtau:.1f}_ntheta256.txt"
        if os.path.exists(filename_S):
            data = np.loadtxt(filename_S)
            # s t a adot exprate tstar mdsqr n e pl T mu
            t_S = data[:,1]
            e_S = data[:,8]
            pL_S = data[:,9]
            T_S = data[:,10]
            mask = t_S * omega_S <= 2.5 * 2*np.pi
            t_S, e_S, pL_S, T_S = t_S[mask], e_S[mask], pL_S[mask], T_S[mask]
            pi_norm_S = pL_S/(e_S/3) -1
            yfac_S = 16/3*eta_s/T0*A*omega_S
            # Calculate corresponding xNS axis for EKT time steps
            xNS_S = - (16/15) * (5 * eta_s / T_S) * A * omega_S * np.cos(omega_S * t_S) / (1 + A*np.sin(omega_S*t_S))
            plot_items.append({
                'x': xNS_S / yfac_S, 'y': pi_norm_S / yfac_S, 'idx': 6
            })
        else:
            print("File %s not found" % filename_S)
        # --- Plotting Loop ---
        for item in plot_items:
            idx = item['idx']
            ax.plot(item['x'], item['y'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

        # Formatting
        xmax_val = 1.1 * np.max(np.abs(xNS_ref))
        ax.set_xlim([-xmax_val, xmax_val])
        ax.set_ylim([-xmax_val, xmax_val])
        ax.set_aspect('equal')
        ax.set_title(r'$\omega \tau_\pi = %g$, $A=%g$' % (wtau, A), fontsize=9)
        ax.set_xlabel(r'$\pi_\mathrm{NS}/(P_{eq}\,\mathcal{N})$')

        if i == 0:
            ax.set_ylabel(r'$\pi/(P_{eq}\,\mathcal{N})$')
            ax.legend(fontsize=7, loc='upper left')

    os.makedirs(outputpath, exist_ok=True)
    out_file = outputpath + f'plot_kinetic_phase_space_A{A}.pdf'
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved {out_file}")
# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Paper figures: Sec. 2 (plot_rate) and Sec. 4.1 (macroscopic and kinetic theories)
    os.makedirs("./paper_figures/", exist_ok=True)
    plot_expansion_rate()                                       # fig:expansion_rate plot_rate.pdf
    plot_MIS_temperature_evolution(A=0.1)                       # fig:MIS            plot_MIS_T_A0.1.pdf
    plot_phase_space_comparison(A=0.1, load_external_data=False)  # fig:MIS2         plot_phase_space_A0.1.pdf
    plot_kinetic_temperature_evolution(A=0.1)                   # fig:kinetic        plot_kinetic_T_A0.1.pdf
    plot_kinetic_phase_space_comparison(A=0.1)                  # fig:kinetic_ps     plot_kinetic_phase_space_A0.1.pdf
