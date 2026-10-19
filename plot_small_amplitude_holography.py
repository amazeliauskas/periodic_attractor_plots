import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Import physics model
import hydro_holography as phys

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
    "#F032E6",  # Magenta (High Contrast vs Red/Purple)
    "#FFE119"   # Yellow
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
LABELS = ['Navier-Stokes', 'Müller-Israel-Stewart', 'MIS kappa=0', 'linear MIS', 'Yang-Mills', 'RTA', 'Scalar', 'Holography']
STYLE = ['-', '-', ':', ':', '-.',':', ':', '--'] 

# Global Plot Settings
FRAME_WIDTH = 2.5
GOLDEN_RATIO = 0.75

# ==========================================
# Simulation Parameters
# ==========================================

# Constants
ETA_S = 1/4/np.pi
T0 = 1/np.pi
WTAU_LIST = [0.1, 0.3, 0.5, 1] # List of omega*tau_pi values to iterate over
KAPPAT_ETA = 1/np.pi
TAU_HOLO = (4-2*np.log(2))*ETA_S/T0
# ==========================================
# Plotting Functions
# ==========================================


def plot_holo_temperature_evolution(A, outputpath="./paper_figures/",
                                    ylim_T=(0.9996, 1.0009),
                                    ylim_pi=(-1.1, 1.1)):
    """
    Plots Temperature (T/T0) and normalised Shear Stress vs Time.
    Glued version: panels share x-axis (col) AND y-axis (row).
    """
    scale = 0.8

    n_plots = len(WTAU_LIST)
    fig, axs = plt.subplots(2, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, scale*2 * FRAME_WIDTH), 
                            sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    if n_plots == 1: axs = axs.reshape(2, 1)

    for i, wtau in enumerate(WTAU_LIST):
        omega = wtau / TAU_HOLO
        t_max_default = 2.5 * (2*np.pi/omega)
        t_eval = np.linspace(0, t_max_default, 1000) # Default t_eval if EKT missing
        xfac = 2*np.pi
        yfac = 16/3*ETA_S/T0*A*omega
        xmax = t_max_default*omega / xfac
        
        t_max = t_max_default
        # List to hold the data sets to plot.
        # Only populated if the data is successfully loaded/calculated.
        plot_items = []

        # NS solution (reference)
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, ETA_S), [0, t_max_default], y0_NS, t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
        pi_NS_norm = - (16/15) * (5*ETA_S/T_NS) * phys.expansion_rate(sol_NS.t, A, omega)

        plot_items.append({
            'x': sol_NS.t * omega/xfac, 'y_T': T_NS/T0, 'y_pi': pi_NS_norm/yfac, 'idx': 0
        })

        # 3. MIS ODE
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S, KAPPAT_ETA), [0, t_max], y0_MIS, t_eval=t_eval)
        T_MIS = phys.findT(sol_MIS.y[0])
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0])

        plot_items.append({
            'x': sol_MIS.t * omega/xfac, 'y_T': T_MIS/T0, 'y_pi': pi_MIS_norm/yfac, 'idx': 1
        })
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S, 0), [0, t_max], y0_MIS, t_eval=t_eval)
        T_MIS = phys.findT(sol_MIS.y[0])
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0])

        plot_items.append({
            'x': sol_MIS.t * omega/xfac, 'y_T': T_MIS/T0, 'y_pi': pi_MIS_norm/yfac, 'idx': 2
        })
        filename = f"./data/holography/small_amplitude/holography_A{A:g}_wtaupi{wtau}.dat"
        if os.path.exists(filename):
            data = np.loadtxt(filename)
            t_holo, pi_NS_holo_v0, pi_holo_v0, T_holo = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            pi_norm_ekt = pi_holo_v0*wtau

            plot_items.append({
                'x': t_holo * omega/xfac, 'y_T': T_holo/T0, 'y_pi': pi_norm_ekt/yfac, 'idx': 7
            })
        else:
            print("File %s not found" % filename)

        # --- Top Row (Temperature T/T0) ---
        ax_T = axs[0, i]
        ax_T.hlines(1, 0, xmax, 'k', lw=0.5)

        # Loop over the populated items to plot
        for item in plot_items:
            idx = item['idx']
            ax_T.plot(item['x'], item['y_T'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

        ax_T.set_title(r'$\omega \tau_\pi = %g$, $A=%g$' % (wtau, A), fontsize=9)
        ax_T.set_ylim(ylim_T)
        ax_T.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useOffset=False))
        if i == 0:
            ax_T.set_ylabel(r'$T/T_0$')
            ax_T.legend(fontsize=7)

        # --- Bottom Row (Shear difference from NS) ---
        ax_Pi = axs[1, i]
        ax_Pi.hlines(0, 0, xmax, 'k', lw=0.5)

        # Loop over the populated items to plot
        for item in plot_items:
            idx = item['idx']
            ax_Pi.plot(item['x'], item['y_pi'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

        ax_Pi.set_xlabel(r'$\omega t/(2\pi)$')
        ax_Pi.set_xlim([0, xmax])
        ax_Pi.set_ylim(ylim_pi)
        if i == 0:
            ax_Pi.set_ylabel(r'$\pi/(P_{eq}\,\mathcal{N})$')

    os.makedirs(outputpath, exist_ok=True)
    plt.savefig(outputpath + f'plot_holo_T_A{A}.pdf', bbox_inches='tight')
    print(f"Saved plot_holo_T_A{A}.pdf")

def plot_holo_temperature_evolution_NSsub(A, wtau=0.1, outputpath="./paper_figures/"):
    """
    Plots normalised Shear Stress difference from NS as a single panel
    at frequency omega*tau_pi = wtau (the paper shows wtau = 0.1).
    """
    scale = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(scale * FRAME_WIDTH, scale * FRAME_WIDTH))

    omega = wtau / TAU_HOLO
    t_max_default = 2.5 * (2*np.pi/omega)
    t_eval = np.linspace(0, t_max_default, 1000)
    xfac = 2*np.pi
    yfac = 16/3*ETA_S/T0*A*omega
    xmax = t_max_default*omega / xfac

    plot_items = []

    # NS solution (reference) -- subtracted from everything else
    y0_NS = [0]
    sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, ETA_S), [0, t_max_default], y0_NS, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    scale_fac = phys.scale_factor(sol_NS.t, A, omega)
    T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
    pi_NS_norm = - (16/15) * (5*ETA_S/T_NS) * phys.expansion_rate(sol_NS.t, A, omega)

    # MIS ODE (kappa from KAPPAT_ETA)
    y0_MIS = [phys.Eeq(T0), 0]
    sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S, KAPPAT_ETA), [0, t_max_default], y0_MIS, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    T_MIS = phys.findT(sol_MIS.y[0])
    pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0])

    plot_items.append({
        'x': sol_MIS.t * omega/xfac, 'y_pi': (pi_MIS_norm - pi_NS_norm)/yfac, 'idx': 1
    })

    # MIS ODE (kappa = 0)
    y0_MIS = [phys.Eeq(T0), 0]
    sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S, 0), [0, t_max_default], y0_MIS, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    T_MIS = phys.findT(sol_MIS.y[0])
    pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0])

    plot_items.append({
        'x': sol_MIS.t * omega/xfac, 'y_pi': (pi_MIS_norm - pi_NS_norm)/yfac, 'idx': 2
    })

    # Holography -- different time grid, so interpolate the NS reference onto it
    filename = f"./data/holography/small_amplitude/holography_A{A:g}_wtaupi{wtau}.dat"
    if os.path.exists(filename):
        data = np.loadtxt(filename)
        t_holo, pi_NS_holo_v0, pi_holo_v0, T_holo = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        pi_norm_ekt = pi_holo_v0*wtau
        pi_NS_norm_holo = np.interp(t_holo, sol_NS.t, pi_NS_norm)

        plot_items.append({
            'x': t_holo * omega/xfac, 'y_pi': (pi_norm_ekt - pi_NS_norm_holo)/yfac, 'idx': 7
        })
    else:
        print("File %s not found" % filename)

    ax.hlines(0, 0, xmax, 'k', lw=0.5)
    for item in plot_items:
        idx = item['idx']
        ax.plot(item['x'], item['y_pi'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

    ax.set_title(r'$\omega \tau_\pi = %g$, $A=%g$' % (wtau, A), fontsize=9)
    ax.set_xlabel(r'$\omega t/(2\pi)$')
    ax.set_xlim([0, xmax])
    ax.set_ylim(-0.2, 0.2)
    ax.set_ylabel(r'$(\pi-\pi_\mathrm{NS})/(P_{eq}\,\mathcal{N})$')
    ax.legend(fontsize=7)

    os.makedirs(outputpath, exist_ok=True)
    plt.savefig(outputpath + f'plot_holo_T_NSsub_A{A}.pdf', bbox_inches='tight')
    print(f"Saved plot_holo_T_NSsub_A{A}.pdf")
    plt.close(fig)

def plot_holo_phase_space_comparison(A, outputpath="./paper_figures/"):
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
        omega = wtau / TAU_HOLO
        t_max_default = 2.5 * (2*np.pi/omega)
        t_eval = np.linspace(0, t_max_default, 1000)

        # Scaling factor for the axes

        yfac = 16/3*ETA_S/T0*A*omega
        plot_items = []

        # 0. Navier-Stokes (Reference)
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, ETA_S), [0, t_max_default], y0_NS, t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
        xNS_ref = - (16/15) * (5*ETA_S/T_NS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac
        
        # 1. Non-linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S, KAPPAT_ETA), [0, t_max_default], y0_MIS, t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        
        plot_items.append({
            'x': xNS_ref, 'y': pi_MIS_norm, 'idx': 1
        })
        # 1. Non-linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, ETA_S, 0), [0, t_max_default], y0_MIS, t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        
        plot_items.append({
            'x': xNS_ref, 'y': pi_MIS_norm, 'idx': 2
        })
        filename = f"./data/holography/small_amplitude/holography_A{A:g}_wtaupi{wtau}.dat"
        if os.path.exists(filename):
            data = np.loadtxt(filename)
            t_holo, pi_NS_holo_v0, pi_holo_v0, T_holo = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            mask = t_holo * omega <= 2.5 * 2*np.pi
            t_holo, pi_holo_v0, T_holo = t_holo[mask], pi_holo_v0[mask], T_holo[mask]
            pi_norm_holo = pi_holo_v0*wtau
            xNS_holo = - (16/15) * (5 * ETA_S / T_holo) * A * omega * np.cos(omega * t_holo) / (1 + A*np.sin(omega*t_holo))
            
            plot_items.append({
                'x': xNS_holo / yfac, 'y': pi_norm_holo / yfac, 'idx': 7
            })           
        else:
            print("File %s not found" % filename)
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
    out_file = outputpath + f'plot_holo_phase_space_A{A}.pdf'
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved {out_file}")
# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Paper figures, Sec. 4.1 (holography)
    os.makedirs("./paper_figures/", exist_ok=True)
    plot_holo_temperature_evolution(A=1e-3)        # fig:holo       plot_holo_T_A0.001.pdf
    plot_holo_temperature_evolution_NSsub(A=1e-3)  # fig:holot_Tsub plot_holo_T_NSsub_A0.001.pdf
    plot_holo_phase_space_comparison(A=1e-3)       # fig:holo_ps    plot_holo_phase_space_A0.001.pdf
