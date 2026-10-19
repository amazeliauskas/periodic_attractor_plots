import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from scipy.signal import medfilt
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
    "#FFA500",
    "#F032E6",  # Magenta (High Contrast vs Red/Purple)
    "#9A6324"   # Deep Brown
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
LABELS = ['Navier-Stokes', 'Müller-Israel-Stewart', 'Free-Streaming', 'Linear MIS*', 'Yang-Mills', 'RTA', 'Scalar', 'rBRSSS']
STYLE = ['-', '--', ':', '-.', '-.',':', ':', '-']
# Global Plot Settings
FRAME_WIDTH = 2.5
GOLDEN_RATIO = 0.75

# ==========================================
# Simulation Parameters
# ==========================================

# Constants
ETA_S = 0.08
T0 = 1.0
WTAU_LIST = [0.1, 0.5, 2.0, 10.0] # List of omega*tau_pi values to iterate over

# ==========================================
# Plotting Functions
# ==========================================

def plot_large_times(A, outputpath="./paper_figures/"):
    """
    Plots the Phase Space diagram: Shear Stress (pi) vs Navier-Stokes Shear (pi_NS).
    Uses the data loading structure from plot_kinetic_temperature_evolution,
    allowing dynamically loaded YM and RTA data to be plotted alongside ODE solutions.
    """
    scale = 0.8

    with plt.rc_context({
    'lines.linewidth': 0.4, 
    'lines.markersize': 1.5, 
    'font.size': 11,                                                            
    'xtick.minor.visible': True,                                                
    'ytick.minor.visible': True,                                                
    'axes.linewidth': 0.5,                                                      
    'axes.labelsize': 11,                                                       
    'legend.fontsize': 10,                                                      
    'xtick.labelsize': 11,                                                      
    'ytick.labelsize': 11 }):
        
        w = 8.0
        # Glued version: panels share y-axis and have zero horizontal space
        fig, ax = plt.subplots(1, 1, figsize=(scale * FRAME_WIDTH, scale * FRAME_WIDTH),
                                sharey=True, gridspec_kw={'wspace': 0})

        eta_s = ETA_S
        tauR0 = 5 * eta_s / T0
        tauPiScalar = 6.10517 * eta_s / T0
        omega = w
        wtau = w * tauR0
        omegaScalar = wtau / tauPiScalar
        t_max_default = 8000/omega
        t_eval = np.linspace(0, t_max_default, 10000000)

        # Scaling factor for the axes
        yfac = 1

        plot_items = []

        # 0. Navier-Stokes (Reference)
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, eta_s), [0, t_max_default], y0_NS,
                        t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac ** (1 / 3))
        xNS_ref = - (16 / 15) * (5 * eta_s / T_NS) * phys.expansion_rate(sol_NS.t, A, omega) /yfac

        plot_items.append({
            'x': xNS_ref, 'y': xNS_ref, 'idx': 0
        })

        # 5. RTA (External Data)
        filename_RTA = f"./data/RTA/long_time/RTA_etas0.08_A0.1_omega8.0.txt"
        if os.path.exists(filename_RTA):
            data = np.loadtxt(filename_RTA)
            t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS_RTA, a_RTA, adot_RTA = data.T
            pi_norm_RTA = 2 / 3 * (pL_RTA - pT_RTA) / (e_RTA / 3)

            # Calculate corresponding xNS axis for RTA time steps
            xNS_RTA = - (16 / 15) * (5 * eta_s / T_RTA) * A * omega * np.cos(omega * t_RTA) / (
                        1 + A * np.sin(omega * t_RTA))

            plot_items.append({
                'wt': omega*t_RTA, 'x': xNS_RTA / yfac, 'y': pi_norm_RTA / yfac, 'idx': 5
            })
        else:
            print("File %s not found" % filename_RTA)

        # --- Plotting Loop ---
        for item in plot_items:
            idx = item['idx']
            
            # Spezialfall für Müller-Israel-Stewart (Spirale mit Farbverlauf)
            if (idx == 1 or idx == 5) and 'wt' in item:
                x = item['x']
                y = item['y']
                wt = item['wt']
                
                # WICHTIG: Schrittweite (Stride), um die Datenmenge zu bändigen!
                # Jedes 100. Element reicht völlig für eine saubere Linie.
                stride = 1
                x_sub = x[::stride]
                y_sub = y[::stride]
                wt_sub = wt[::stride]
                
                # Segmente für LineCollection bauen
                points = np.array([x_sub, y_sub]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Farbverlauf wählen (z.B. 'viridis', 'plasma', 'ocean' oder 'YlGnBu')
                cmap = plt.get_cmap('viridis') 
                norm = plt.Normalize(wt_sub.min(), wt_sub.max())
                
                # LineCollection erstellen mit leichter Transparenz (alpha)
                lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.7)
                lc.set_array(wt_sub)
                lc.set_linewidth(1.0) # Schön dünne Linie, damit man das Spiralelement sieht
                lc.set_rasterized(True)  # 166k coloured segments: store as bitmap (savefig dpi) to keep the PDF small
                
                # Linie zum Plot hinzufügen
                line = ax.add_collection(lc)

                ax.plot([], [], color=cmap(0.5), ls='-', label=LABELS[idx])
                
                # Optional: Eine Farblegende (Colorbar) an der Seite hinzufügen
                cbar = fig.colorbar(line, ax=ax, pad=0.03, shrink=0.7)
                cbar.set_label(r'$\omega t$', fontsize=4)
                cbar.ax.tick_params(labelsize=4)
                
            else:
                # Standard-Plot für Navier-Stokes und andere Referenzen
                ax.plot(item['x'], item['y'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

        # Formatting
        xmax_val = 1.1 * np.max(np.abs(xNS_ref))
        ax.set_xlim([-xmax_val, xmax_val])
        ax.set_ylim([-xmax_val, xmax_val])
        ax.set_aspect('equal')
        ax.set_title(r'$A=%g,$ $\omega \tau_\pi = %g$' % (A, wtau), fontsize=4)
        ax.set_xlabel(r'$\pi_{NS}/P_{\text{eq}}$', fontsize=4)
        ax.tick_params(labelsize=4)

        ax.set_ylabel(r'$\pi/P_{\text{eq}}$', fontsize=4)
        ax.legend(fontsize=4, loc='upper left')

        os.makedirs(outputpath, exist_ok=True)
        out_file = outputpath + f'plot_RTA_large_times.pdf'
        plt.savefig(out_file, dpi = 500, transparent=True, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_file}")


def plot_Large_Scan_rBRSSS(outputpath="./paper_figures/"):
    """
    Plots the Phase Space diagram: Shear Stress (pi) vs Navier-Stokes Shear (pi_NS).
    Uses the data loading structure from plot_kinetic_temperature_evolution, 
    allowing dynamically loaded YM and RTA data to be plotted alongside ODE solutions.
    """
    scale = 0.8
    A_LIST = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    w1_LIST = np.array([0.47, 1.18, 2.35, 4.70, 11.57])
    w2_LIST = np.array([0.23, 0.58, 1.16, 2.31, 5.97])
    w3_LIST = np.array([0.15, 0.39, 0.77, 1.54, 3.89])
    w4_LIST = np.array([0.11, 0.28, 0.57, 1.13, 2.85])
    w5_LIST = np.array([0.09, 0.22, 0.44, 0.89, 2.27])
    w6_LIST = np.array([0.07, 0.18, 0.35, 0.70, 1.81])
    w7_LIST = np.array([0.06, 0.15, 0.29, 0.58, 1.53])
    w8_LIST = np.array([0.05, 0.12, 0.24, 0.48, 1.24])
    w9_LIST = np.array([0.04, 0.09, 0.19, 0.38, 0.99])

    maskeA = [True, False, True, False, True, False, True, False, True]
    maskew = [False, True, True, True, True]

    A_LIST = A_LIST[maskeA]
    w1_LIST = w1_LIST[maskew]
    w2_LIST = w2_LIST[maskew]
    w3_LIST = w3_LIST[maskew]
    w4_LIST = w4_LIST[maskew]
    w5_LIST = w5_LIST[maskew]
    w6_LIST = w6_LIST[maskew]
    w7_LIST = w7_LIST[maskew]
    w8_LIST = w8_LIST[maskew]
    w9_LIST = w9_LIST[maskew]

    m_plots = len(A_LIST)
    n_plots = len(w1_LIST)

    w_LISTS = np.array([[1.25, 2.49, 4.99, 12.47],[0.41, 0.82, 1.63, 4.08],[0.23, 0.47, 0.93, 2.33],[0.15, 0.30, 0.60, 1.51],[0.09, 0.19, 0.38, 0.94]])
    Knudsen = [0.05, 0.1, 0.2, 0.5]
    # Glued version: panels share y-axis and have zero horizontal space
    fig, axs = plt.subplots(m_plots, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, m_plots*scale*FRAME_WIDTH), gridspec_kw={'wspace': 0.3})
    
    if n_plots == 1 and m_plots == 1: axs = [axs]

    def add_info(ax, w, A):
        textstr = rf"$A = {A}$, $\omega \tau_\pi= {w}$"

        ax.text(
            0.95, 0.05,
            textstr,
            transform = ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    for i, A in enumerate(A_LIST):
        for j, w in enumerate(w_LISTS[i]):
            ax = axs[i, j]
            if(A == 0.1):
                ax.set_title(rf'$\text{{Kn}}_{{\text{{max}}}} = {Knudsen[j]}$', fontsize=12, pad=10)
            eta_s = ETA_S
            tauR0 = 5 * eta_s / T0
            tauPiScalar = 6.10517 * eta_s / T0
            omega = w
            wtau = w*tauR0
            omegaScalar = wtau / tauPiScalar
            N_PERIODS = 4   # same time window for NS, rBRSSS and RTA
            t_max_default = N_PERIODS * (2*np.pi/omega)
            t_eval = np.linspace(0, t_max_default, 1000)

            # Scaling factor for the axes
            yfac = 1
            
            plot_items = []

            # 0. Navier-Stokes (Reference)
            y0_NS = [0]
            sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, eta_s), [0, t_max_default], y0_NS, t_eval=t_eval)
            scale_fac = phys.scale_factor(sol_NS.t, A, omega)
            T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
            xNS_ref = - (16/15) * (5*eta_s/T_NS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac
            
            plot_items.append({
                'x': xNS_ref, 'y': xNS_ref, 'idx': 0
            })

            # 1. rBRSSS
            y0_MIS = [phys.Eeq(T0), 0]
            sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, "lambda"), [0, t_max_default], y0_MIS, t_eval=t_eval)
            pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
            
            plot_items.append({
                'x': xNS_ref, 'y': pi_MIS_norm, 'idx': 7
            })

            # 2. Free Streaming
            a_val = phys.scale_factor(t_eval, A, omega)
            pi_free_norm = phys.Pi_P_free_streaming(a_val) / yfac

            # 5. RTA (External Data)
            filename_RTA = f"./data/RTA/nonlinear/RTA_etas0.08_A%g_omega%g.txt" % (A, w)
            
            if os.path.exists(filename_RTA):
                data = np.loadtxt(filename_RTA)
                t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS_RTA, a_RTA, adot_RTA, wt_RTA = data.T
                mask = t_RTA <= t_max_default   # RTA files run to omega t = 100 (~16 periods)
                t_RTA, e_RTA, pT_RTA, pL_RTA, T_RTA = t_RTA[mask], e_RTA[mask], pT_RTA[mask], pL_RTA[mask], T_RTA[mask]
                pi_norm_RTA = 2/3*(pL_RTA-pT_RTA)/(e_RTA/3) 
                
                # Calculate corresponding xNS axis for RTA time steps
                xNS_RTA = - (16/15) * (5 * eta_s / T_RTA) * A * omega * np.cos(omega * t_RTA) / (1 + A*np.sin(omega*t_RTA)) 
                plot_items.append({
                    'x': xNS_RTA / yfac, 'y': pi_norm_RTA / yfac, 'idx': 5
                })
            else:
                print("File %s not found" % filename_RTA)

            # --- Plotting Loop ---
            for item in plot_items:
                idx = item['idx']
                ax.plot(item['x'], item['y'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

            # Formatting
            xmax_val = 1.4
            add_info(ax, round(wtau, 2), A)

            if j == 0:
                ax.set_ylabel(r'$\pi/P_{\text{eq}}$')

            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='upper left')

            if i == m_plots-1:
                ax.set_xlabel(r'$\pi_{NS}/P_{\text{eq}}$')

    os.makedirs(outputpath, exist_ok=True)
    out_file = outputpath + f'plot_Large_Scan_Non_Linear.pdf'
    plt.savefig(out_file, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

def plot_MIS_rBRSSS_PI(method="MIS", outputpath="./paper_figures/"):
    """
    method="MIS"    -> RTA vs MIS and Navier-Stokes    (plot_nonlin_ps_MIS.pdf)
    method="lambda" -> RTA vs rBRSSS and Navier-Stokes (plot_nonlin_ps_rBRSSS.pdf)
    Plots the Phase Space diagram: Shear Stress (pi) vs Navier-Stokes Shear (pi_NS).
    Uses the data loading structure from plot_kinetic_temperature_evolution, 
    allowing dynamically loaded YM and RTA data to be plotted alongside ODE solutions.
    """
    scale = 0.8
    A_LIST = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    w1_LIST = np.array([0.47, 1.18, 2.35, 4.70, 11.57])
    w2_LIST = np.array([0.23, 0.58, 1.16, 2.31, 5.97])
    w3_LIST = np.array([0.15, 0.39, 0.77, 1.54, 3.89])
    w4_LIST = np.array([0.11, 0.28, 0.57, 1.13, 2.85])
    w5_LIST = np.array([0.09, 0.22, 0.44, 0.89, 2.27])
    w6_LIST = np.array([0.07, 0.18, 0.35, 0.70, 1.81])
    w7_LIST = np.array([0.06, 0.15, 0.29, 0.58, 1.53])
    w8_LIST = np.array([0.05, 0.12, 0.24, 0.48, 1.24])
    w9_LIST = np.array([0.04, 0.09, 0.19, 0.38, 0.99])

    maskeA = [False, False, True, False, False, False, True, False, False]
    maskew = [False, False, False, True, True]

    A_LIST = A_LIST[maskeA]
    w1_LIST = w1_LIST[maskew]
    w2_LIST = w2_LIST[maskew]
    w3_LIST = w3_LIST[maskew]
    w4_LIST = w4_LIST[maskew]
    w5_LIST = w5_LIST[maskew]
    w6_LIST = w6_LIST[maskew]
    w7_LIST = w7_LIST[maskew]
    w8_LIST = w8_LIST[maskew]
    w9_LIST = w9_LIST[maskew]

    m_plots = len(A_LIST)
    n_plots = len(w1_LIST)

    w_LISTS = np.array([w1_LIST, w2_LIST, w3_LIST, w4_LIST, w5_LIST, w6_LIST, w7_LIST, w8_LIST, w9_LIST])[maskeA]
    w_LISTS = np.array([[1.63, 4.08],[0.60, 1.51]])
    # Glued version: panels share y-axis and have zero horizontal space
    fig, axs = plt.subplots(m_plots, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, m_plots*scale*FRAME_WIDTH), gridspec_kw={'wspace': 0.3})
    
    if n_plots == 1 and m_plots == 1: axs = [axs]

    def add_info(ax, w, A):
        textstr = rf"$A = {A}$, $\omega \tau_\pi = {w}$"

        ax.text(
            0.95, 0.05,
            textstr,
            transform = ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    for i, A in enumerate(A_LIST):
        for j, w in enumerate(w_LISTS[i]):
            ax = axs[i, j]
            eta_s = ETA_S
            tauR0 = 5 * eta_s / T0
            tauPiScalar = 6.10517 * eta_s / T0
            omega = w
            wtau = w*tauR0
            omegaScalar = wtau / tauPiScalar
            N_PERIODS = 4   # same time window for NS, MIS/rBRSSS and RTA
            t_max_default = N_PERIODS * (2*np.pi/omega)
            t_eval = np.linspace(0, t_max_default, 1000)

            # Scaling factor for the axes
            yfac = 1
            
            plot_items = []

            # 0. Navier-Stokes (Reference)
            y0_NS = [0]
            sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, eta_s), [0, t_max_default], y0_NS, t_eval=t_eval)
            scale_fac = phys.scale_factor(sol_NS.t, A, omega)
            T_NS = (T0 + sol_NS.y[0]) / (scale_fac**(1/3))
            xNS_ref = - (16/15) * (5*eta_s/T_NS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac
            
            plot_items.append({
                'x': xNS_ref, 'y': xNS_ref, 'idx': 0
            })

            # 1. Non-linear MIS (method="MIS") or rBRSSS (method="lambda")
            y0_MIS = [phys.Eeq(T0), 0]
            sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, method), [0, t_max_default], y0_MIS, t_eval=t_eval)
            pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac

            plot_items.append({
                'x': xNS_ref, 'y': pi_MIS_norm, 'idx': 1 if method == "MIS" else 7
            })

            # 2. Free Streaming
            a_val = phys.scale_factor(t_eval, A, omega)
            pi_free_norm = phys.Pi_P_free_streaming(a_val) / yfac

            # 5. RTA (External Data)

            filename_RTA = f"./data/RTA/nonlinear/RTA_etas0.08_A%g_omega%g.txt" % (A, w)
            
            if os.path.exists(filename_RTA):
                data = np.loadtxt(filename_RTA)
                t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS_RTA, a_RTA, adot_RTA, wt_RTA = data.T
                mask = t_RTA <= t_max_default   # RTA files run to omega t = 100 (~16 periods)
                t_RTA, e_RTA, pT_RTA, pL_RTA, T_RTA = t_RTA[mask], e_RTA[mask], pT_RTA[mask], pL_RTA[mask], T_RTA[mask]
                pi_norm_RTA = 2/3*(pL_RTA-pT_RTA)/(e_RTA/3) 
                
                # Calculate corresponding xNS axis for RTA time steps
                xNS_RTA = - (16/15) * (5 * eta_s / T_RTA) * A * omega * np.cos(omega * t_RTA) / (1 + A*np.sin(omega*t_RTA))

                plot_items.append({
                    'x': xNS_RTA / yfac, 'y': pi_norm_RTA / yfac, 'idx': 5
                })
            else:
                print("File %s not found" % filename_RTA)

            # --- Plotting Loop ---
            for item in plot_items:
                idx = item['idx']
                ax.plot(item['x'], item['y'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

            # Formatting
            xmax_val = 1.4
            add_info(ax, round(wtau, 2), A)

            if j == 0:
                ax.set_ylabel(r'$\pi/P_{\text{eq}}$')

            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='upper left')

            if i == m_plots-1:
                ax.set_xlabel(r'$\pi_{NS}/P_{\text{eq}}$')

    os.makedirs(outputpath, exist_ok=True)
    out_file = outputpath + ('plot_nonlin_ps_MIS.pdf' if method == "MIS" else 'plot_nonlin_ps_rBRSSS.pdf')
    plt.savefig(out_file, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")


def plot_MIS_rBRSSS_temperature(outputpath="./paper_figures/"):
    """
    Plots Temperature and Shear Stress vs Time.
    Glued version: panels share x-axis (col) AND y-axis (row).
    """
    scale = 0.8
    dataset = [[0.3, 1.63],[0.3, 4.08],[0.7, 0.60],[0.7, 1.51]]
    n_plots = 4
    fig, axs = plt.subplots(2, n_plots, figsize=(scale*n_plots * FRAME_WIDTH, scale*2 * FRAME_WIDTH), 
                            sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    if n_plots == 1: axs = axs.reshape(2, 1)

    for i, Aomega in enumerate(dataset):
        A = Aomega[0]
        omega = Aomega[1]
        eta_s = 0.08
        tauR0 = 5 * eta_s / T0
        w = omega
        wtau = w*tauR0
        t_max_default = 3.5 * (2*np.pi/omega)
        t_eval = np.linspace(0, t_max_default, 1000) # Default t_eval if EKT missing
        xfac = 2*np.pi
        xmax = t_max_default*omega / xfac
        
        yfac = 1

        t_max = t_max_default
        # List to hold the data sets to plot. 
        # Only populated if the data is successfully loaded/calculated.
        plot_items = []
        
        # 0. Navier-Stokes (Reference)
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, eta_s), [0, t_max_default], y0_NS,
                        t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac ** (1 / 3))
        xNS_ref = - (16 / 15) * (5 * eta_s / T_NS) * phys.expansion_rate(sol_NS.t, A, omega) /yfac

        plot_items.append({
            'wt': omega*t_eval/(2*np.pi), 'x': xNS_ref, 'y': xNS_ref, 'T': T_NS, 'idx': 0
        })

        # 1. Non-linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, "MIS"), [0, t_max_default], y0_MIS,
                            t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        T_MIS = phys.findT(sol_MIS.y[0]) / yfac
        xNS_MIS = - (16 / 15) * (5 * eta_s / T_MIS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac

        plot_items.append({
            'wt': omega*t_eval/(2*np.pi), 'x': xNS_MIS, 'y': pi_MIS_norm, 'T': T_MIS, 'idx': 1
        })

        # 7. rBRSSS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, "lambda"), [0, t_max_default], y0_MIS,
                            t_eval=t_eval)
        pi_rBRSSS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        T_rBRSSS = phys.findT(sol_MIS.y[0]) / yfac
        xNS_rBRSSS = - (16 / 15) * (5 * eta_s / T_MIS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac

        plot_items.append({
            'wt': omega*t_eval/(2*np.pi), 'x': xNS_rBRSSS, 'y': pi_rBRSSS_norm, 'T': T_rBRSSS, 'idx': 7
        })

        # 5. RTA (External Data)
        filename_RTA = f"./data/RTA/nonlinear/RTA_etas0.08_A%g_omega%g.txt" % (A, w)
        if os.path.exists(filename_RTA):
            data = np.loadtxt(filename_RTA)
            t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS_RTA, a_RTA, adot_RTA, wt_RTA = data.T
            pi_norm_RTA = 2 / 3 * (pL_RTA - pT_RTA) / (e_RTA / 3)

            # Calculate corresponding xNS axis for RTA time steps
            xNS_RTA = - (16 / 15) * (5 * eta_s / T_RTA) * A * omega * np.cos(omega * t_RTA) / (
                        1 + A * np.sin(omega * t_RTA))

            plot_items.append({
                'wt': omega*t_RTA/(2*np.pi), 'x': xNS_RTA / yfac, 'y': pi_norm_RTA / yfac, 'T': T_RTA, 'idx': 5
            })
        else:
            print("File %s not found" % filename_RTA)

        # --- Top Row (Temperature) ---
        ax_T = axs[0, i]
        ax_T.hlines(1, 0, xmax, 'k', lw=0.5)
        
        # Loop over the populated items to plot
        for item in plot_items:
            idx = item['idx']
            ax_T.plot(item['wt'], item['T'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])
        
        ax_T.set_title(r'$A=%g$, $\omega \tau_\pi = %g$' % (A, wtau), fontsize=9)
        if i == 0: 
            ax_T.set_ylabel(r'$T/T_0$')
            ax_T.legend(fontsize=7)

        # --- Bottom Row (Shear) ---
        ax_Pi = axs[1, i]
        ax_Pi.hlines(0, 0, xmax, 'k', lw=0.5)
        
        # Loop over the populated items to plot
        for item in plot_items:
            idx = item['idx']
            ax_Pi.plot(item['wt'], item['y'], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])

        ax_Pi.set_xlabel(r'$\omega t/(2\pi)$')
        ax_Pi.set_xlim([0, xmax])
        if i == 0: 
            ax_Pi.set_ylabel(r'$\pi/P$')

    os.makedirs(outputpath, exist_ok=True)
    plt.savefig(outputpath + 'plot_MIS_rBRSSS.pdf', transparent=True, bbox_inches='tight')
    print("Saved plot_MIS_rBRSSS.pdf")

def plot_large_times_PI(A, outputpath="./paper_figures/"):
    """
    Plots the Phase Space diagram: Shear Stress (pi) vs Navier-Stokes Shear (pi_NS).
    Uses the data loading structure from plot_kinetic_temperature_evolution,
    allowing dynamically loaded YM and RTA data to be plotted alongside ODE solutions.
    """
    
    scale = 0.8

    with plt.rc_context({
     'lines.linewidth': 0.4, 
     'lines.markersize': 1.5, 
     'font.size': 11,                                                            
     'xtick.minor.visible': True,                                                
     'ytick.minor.visible': True,                                                
     'axes.linewidth': 0.5,                                                      
     'axes.labelsize': 11,                                                       
     'legend.fontsize': 10,                                                      
     'xtick.labelsize': 11,                                                      
     'ytick.labelsize': 11 }):
        w = 8.0

        fig = plt.figure(figsize=(scale * FRAME_WIDTH, scale * FRAME_WIDTH))
        
        # 1. Ein Haupt-Raster mit 2 Zeilen definieren (oben für die großen, unten für die kleinen Plots)
        # height_ratios=[2, 1] sorgt dafür, dass die Proportionen (2/3 zu 1/3) wie vorher bleiben.
        gs_main = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

        # 2. Sub-Raster für den oberen Teil (2 Zeilen, 1 Spalte -> hspace=0.0 für nahtlosen Übergang)
        gs_top = gs_main[0].subgridspec(2, 1, hspace=0.0)
        ax_top = fig.add_subplot(gs_top[0, 0])     # Ganz oben
        ax_middle = fig.add_subplot(gs_top[1, 0])  # Die Mitte (zweiter großer Plot)

        # 3. Sub-Raster für den unteren Teil (1 Zeile, 2 Spalten -> wspace für horizontalen Abstand)
        gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.3)
        ax_bottom_l = fig.add_subplot(gs_bottom[0, 0]) # Unten links
        ax_bottom_r = fig.add_subplot(gs_bottom[0, 1], sharey=ax_bottom_l) # Unten rechts

        fig.align_ylabels([ax_top, ax_middle, ax_bottom_l])
        # 4. X-Achsen-Beschriftung des ganz oberen Plots unsichtbar machen
        # Dadurch kleben sie nahtlos aneinander und nur der mittlere Plot zeigt die Zeitachse.
        ax_top.tick_params(labelbottom=False)

        ax_bottom_l.set_xticks([0, 10, 20])
        ax_bottom_r.set_xticks([9000, 9010, 9020])
        # 5. Deine restlichen Formatierungen für die unteren Plots
        ax_bottom_l.tick_params(axis='both', labelbottom=True, labelsize=4)
        ax_bottom_r.tick_params(axis='both', labelbottom=True, labelsize=4)

        eta_s = ETA_S
        tauR0 = 5 * eta_s / T0
        tauPiScalar = 6.10517 * eta_s / T0
        omega = w
        wtau = w * tauR0
        omegaScalar = wtau / tauPiScalar
        t_max_default = 10000/omega
        t_eval = np.linspace(0, t_max_default, 10000000)

        # Scaling factor for the axes
        yfac = 1

        plot_items = []

        # 0. Navier-Stokes (Reference)
        y0_MIS = [phys.Eeq(T0), 0]
        y0_NS = [0]
        sol_NS = solve_ivp(lambda t, y: phys.dydt_NS_heating(t, y, A, omega, eta_s), [0, t_max_default], y0_NS,
                        t_eval=t_eval)
        scale_fac = phys.scale_factor(sol_NS.t, A, omega)
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, "MIS"), [0, t_max_default], y0_MIS,
                            t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        T_MIS = phys.findT(sol_MIS.y[0]) / yfac
        T_NS = (T0 + sol_NS.y[0]) / (scale_fac ** (1 / 3))
        xNS_ref = - (16 / 15) * (5 * eta_s / T_MIS) * phys.expansion_rate(sol_NS.t, A, omega) /yfac

        indices_upper_PI_NS, _ = find_peaks(xNS_ref, distance=10)
        indices_lower_PI_NS, _ = find_peaks(-xNS_ref, distance=10)  
        indices_upper_T_NS, _ = find_peaks(T_NS, distance=10)
        indices_lower_T_NS, _ = find_peaks(-T_NS, distance=10) 

        t_upper_T_NS = t_eval[indices_upper_T_NS]
        T_NS_upper_NS = T_NS[indices_upper_T_NS]
        t_lower_T_NS = t_eval[indices_lower_T_NS]
        T_NS_lower_NS = T_NS[indices_lower_T_NS]

        t_upper_PI_NS = t_eval[indices_upper_PI_NS]
        PI_upper_NS = xNS_ref[indices_upper_PI_NS]
        t_lower_PI_NS = t_eval[indices_lower_PI_NS]
        PI_lower_NS = xNS_ref[indices_lower_PI_NS]

        plot_items.append({
            'wt_upper_T': omega*t_upper_T_NS, 'wt_upper_PI': omega*t_upper_PI_NS, 'y_upper': PI_upper_NS, 'T_upper': T_NS_upper_NS,
            'wt_lower_T': omega*t_lower_T_NS, 'wt_lower_PI': omega*t_lower_PI_NS, 'y_lower': PI_lower_NS, 'T_lower': T_NS_lower_NS, 'wt': omega*t_eval, 'PI': xNS_ref, 'T': T_NS,'idx': 0
        })

        # 1. Non-linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, "MIS"), [0, t_max_default], y0_MIS,
                            t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        T_MIS = phys.findT(sol_MIS.y[0]) / yfac
        xNS_MIS = - (16 / 15) * (5 * eta_s / T_MIS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac

        indices_upper_PI_MIS, _ = find_peaks(pi_MIS_norm, distance=10)
        indices_lower_PI_MIS, _ = find_peaks(-pi_MIS_norm, distance=10)  
        indices_upper_T_MIS, _ = find_peaks(T_MIS, distance=10)
        indices_lower_T_MIS, _ = find_peaks(-T_MIS, distance=10) 

        T_MIS_upper_MIS = np.maximum.accumulate(T_MIS[indices_upper_T_MIS])
        t_upper_T_MIS = t_eval[indices_upper_T_MIS]
        
        t_lower_T_MIS = t_eval[indices_lower_T_MIS]
        T_MIS_lower_MIS = medfilt(T_MIS[indices_lower_T_MIS], kernel_size=5)

        schwellenwert = 0.05

        t_upper_PI_MIS = t_eval[indices_upper_PI_MIS]
        PI_upper_MIS = pi_MIS_norm[indices_upper_PI_MIS]
        differenzen_upper_MIS = np.diff(PI_upper_MIS)

        maske_upper_MIS = np.ones(len(PI_upper_MIS), dtype=bool)
        maske_upper_MIS[:-1] = np.abs(differenzen_upper_MIS) < schwellenwert

        t_upper_PI_MIS = t_upper_PI_MIS[maske_upper_MIS]
        PI_upper_MIS = PI_upper_MIS[maske_upper_MIS]

        t_lower_PI_MIS = t_eval[indices_lower_PI_MIS]
        PI_lower_MIS = pi_MIS_norm[indices_lower_PI_MIS]
        differenzen_lower_MIS = np.diff(PI_lower_MIS)

        maske_lower_MIS = np.ones(len(PI_lower_MIS), dtype=bool)
        maske_lower_MIS[:-1] = np.abs(differenzen_lower_MIS) < schwellenwert

        t_lower_PI_MIS = t_lower_PI_MIS[maske_lower_MIS]
        PI_lower_MIS = PI_lower_MIS[maske_lower_MIS]

        plot_items.append({
            'wt_upper_T': omega*t_upper_T_MIS, 'wt_upper_PI': omega*t_upper_PI_MIS, 'y_upper': PI_upper_MIS, 'T_upper': T_MIS_upper_MIS,
            'wt_lower_T': omega*t_lower_T_MIS, 'wt_lower_PI': omega*t_lower_PI_MIS, 'y_lower': PI_lower_MIS, 'T_lower': T_MIS_lower_MIS, 'wt': omega*t_eval, 'PI': pi_MIS_norm, 'T': T_MIS, 'idx': 1
        })

        # 3. Linear MIS
        y0_MIS = [phys.Eeq(T0), 0]
        sol_MIS = solve_ivp(lambda t, y: phys.dydt_MIS(t, y, A, omega, eta_s, "MIS"), [0, t_max_default], y0_MIS,
                            t_eval=t_eval)
        pi_MIS_norm = sol_MIS.y[1] / phys.pressure(sol_MIS.y[0]) / yfac
        T_MIS = phys.findT(sol_MIS.y[0]) / yfac
        xNS_MIS = - (16 / 15) * (5 * eta_s / T_MIS) * phys.expansion_rate(sol_NS.t, A, omega) / yfac

        omegat = t_eval*omega
        wtau_dyn = wtau*T0/T_MIS
        phi = np.arctan(wtau_dyn)
        pi_MIS_lin_norm = - (16/15) * A*wtau_dyn * (np.cos(omegat - phi) / np.sqrt(wtau_dyn**2 + 1) - np.exp(-t_eval*omega/wtau_dyn)/(wtau_dyn**2 + 1) ) / yfac
        
        indices_upper_PI_MIS_LIN, _ = find_peaks(pi_MIS_lin_norm, distance=10)
        indices_lower_PI_MIS_LIN, _ = find_peaks(-pi_MIS_lin_norm, distance=10)  
        indices_upper_T_MIS_LIN, _ = find_peaks(T_MIS, distance=10)
        indices_lower_T_MIS_LIN, _ = find_peaks(-T_MIS, distance=10) 

        T_MIS_upper_MIS_LIN = np.maximum.accumulate(T_MIS[indices_upper_T_MIS_LIN])
        t_upper_T_MIS_LIN = t_eval[indices_upper_T_MIS_LIN]
        
        t_lower_T_MIS_LIN = t_eval[indices_lower_T_MIS_LIN]
        T_MIS_lower_MIS_LIN = np.maximum.accumulate(T_MIS[indices_lower_T_MIS_LIN])

        schwellenwert = 0.05

        t_upper_PI_MIS_LIN = t_eval[indices_upper_PI_MIS_LIN]
        PI_upper_MIS_LIN = pi_MIS_lin_norm[indices_upper_PI_MIS_LIN]
        differenzen_upper_MIS_LIN = np.diff(PI_upper_MIS_LIN)

        maske_upper_MIS_LIN = np.ones(len(PI_upper_MIS_LIN), dtype=bool)
        maske_upper_MIS_LIN[:-1] = np.abs(differenzen_upper_MIS_LIN) < schwellenwert

        t_upper_PI_MIS_LIN = t_upper_PI_MIS_LIN[maske_upper_MIS_LIN]
        PI_upper_MIS_LIN = PI_upper_MIS_LIN[maske_upper_MIS_LIN]

        t_lower_PI_MIS_LIN = t_eval[indices_lower_PI_MIS_LIN]
        PI_lower_MIS_LIN = pi_MIS_lin_norm[indices_lower_PI_MIS_LIN]
        differenzen_lower_MIS_LIN = np.diff(PI_lower_MIS_LIN)

        maske_lower_MIS_LIN = np.ones(len(PI_lower_MIS_LIN), dtype=bool)
        maske_lower_MIS_LIN[:-1] = np.abs(differenzen_lower_MIS_LIN) < schwellenwert

        t_lower_PI_MIS_LIN = t_lower_PI_MIS_LIN[maske_lower_MIS_LIN]
        PI_lower_MIS_LIN = PI_lower_MIS_LIN[maske_lower_MIS_LIN]

        plot_items.append({
            'wt_upper_T': omega*t_upper_T_MIS_LIN, 'wt_upper_PI': omega*t_upper_PI_MIS_LIN, 'y_upper': PI_upper_MIS_LIN, 'T_upper': T_MIS_upper_MIS_LIN,
            'wt_lower_T': omega*t_lower_T_MIS_LIN, 'wt_lower_PI': omega*t_lower_PI_MIS_LIN, 'y_lower': PI_lower_MIS_LIN, 'T_lower': T_MIS_lower_MIS_LIN, 'wt': omega*t_eval, 'PI': pi_MIS_lin_norm, 'T': T_MIS, 'idx': 3
        })

        # 5. RTA (External Data)
        filename_RTA = f"./data/RTA/long_time/RTA_etas0.08_A0.1_omega8.0.txt"
        if os.path.exists(filename_RTA):
            data = np.loadtxt(filename_RTA)
            t_RTA, e_RTA, pT_RTA, pL_RTA, rate_RTA, T_RTA, e_NS, P_L_NS, T_NS_RTA, a_RTA, adot_RTA = data.T
            pi_norm_RTA = 2 / 3 * (pL_RTA - pT_RTA) / (e_RTA / 3)
        
            # Calculate corresponding xNS axis for RTA time steps
            xNS_RTA = - (16 / 15) * (5 * eta_s / T_RTA) * A * omega * np.cos(omega * t_RTA) / (1 + A * np.sin(omega * t_RTA))
            indices_upper_PI, _ = find_peaks(pi_norm_RTA, distance=10)
            indices_lower_PI, _ = find_peaks(-pi_norm_RTA, distance=10)  
            indices_upper_T, _ = find_peaks(T_RTA, distance=10)
            indices_lower_T, _ = find_peaks(-T_RTA, distance=10) 

            t_upper_T = t_RTA[indices_upper_T]
            T_RTA_upper = T_RTA[indices_upper_T]
            t_lower_T = t_RTA[indices_lower_T]
            T_RTA_lower = T_RTA[indices_lower_T]

            t_upper_PI = t_RTA[indices_upper_PI]
            PI_upper = pi_norm_RTA[indices_upper_PI]
            t_lower_PI = t_RTA[indices_lower_PI]
            PI_lower = pi_norm_RTA[indices_lower_PI]

            plot_items.append({
                'wt_upper_T': omega*t_upper_T, 'wt_upper_PI': omega*t_upper_PI, 'y_upper': PI_upper, 'T_upper': T_RTA_upper,
                'wt_lower_T': omega*t_lower_T, 'wt_lower_PI': omega*t_lower_PI, 'y_lower': PI_lower, 'T_lower': T_RTA_lower, 'wt': omega*t_RTA, 'PI': pi_norm_RTA, 'T': T_RTA, 'idx': 5
            })
        else:
            print("File %s not found" % filename_RTA)

        ax_PI = ax_middle
        ax_T = ax_top
        
        wt_lower1 = 0
        wt_upper1 = 20
        wt_lower2 = 9000
        wt_upper2 = 9020

        # --- Plotting Loop ---
        for item in plot_items:
            idx = item['idx']
            maske1 = (item['wt'] < wt_upper1) & (item['wt'] >= wt_lower1)
            maske2 = (item['wt'] < wt_upper2) & (item['wt'] >= wt_lower2)
 
            ax_bottom_l.plot(item['wt'][maske1], item['PI'][maske1], color=COLORS[idx], ls=STYLE[idx])
            ax_bottom_r.plot(item['wt'][maske2], item['PI'][maske2], color=COLORS[idx], ls=STYLE[idx])
            if(item['idx']==0):    
                ax_PI.plot(item['wt_lower_PI'], item['y_lower'], color=COLORS[idx], ls=STYLE[idx], label='Navier-Stokes*')
            if(item['idx']==3):    
                ax_PI.plot(item['wt_lower_PI'], item['y_lower'], color=COLORS[idx], ls=STYLE[idx], label='Linear MIS*')
            else:
                ax_PI.plot(item['wt_lower_PI'], item['y_lower'], color=COLORS[idx], ls=STYLE[idx])
            ax_PI.plot(item['wt_upper_PI'], item['y_upper'], color=COLORS[idx], ls=STYLE[idx])
            if(item['idx']==1 or item['idx']==5):
                ax_T.plot([], [], color=COLORS[idx], ls=STYLE[idx], label=LABELS[idx])
                ax_T.plot(item['wt_lower_T'], item['T_lower'], color=COLORS[idx], ls=STYLE[idx])
                ax_T.plot(item['wt_upper_T'], item['T_upper'], color=COLORS[idx], ls=STYLE[idx])

        # Formatting

        ax_bottom_l.set_xlim([wt_lower1,wt_upper1])
        ax_bottom_r.set_xlim([wt_lower2,wt_upper2])
        ax_bottom_l.set_xlabel(r'$\omega t$', fontsize = 4)
        ax_bottom_r.set_xlabel(r'$\omega t$', fontsize = 4)
        ax_bottom_l.set_ylabel(r'$\pi/P_{\text{eq}}$', fontsize = 4)
        ax_bottom_r.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax_T.set_title(r'$A=%g,$ $\omega \tau_\pi = %g$' % (A, wtau), fontsize=4)

        ax_T.set_yticks([0.0, 2.5, 5.0, 7.5])
        ax_T.tick_params(axis='both', labelbottom=False, labelsize=4)

        ax_T.set_xlim([0, 10000])

        ax_PI.tick_params(axis='both', labelbottom=True, labelsize=4)

        ax_PI.set_xlim([0, 10000])
        
        ax_T.set_ylabel(r'$(T/T_0)_{\text{env}}$', fontsize = 4)

        ax_PI.set_xlabel(r'$\omega t$', fontsize = 4, labelpad=2)
        ax_PI.set_ylabel(r'$(\pi/P_{\text{eq}})_{\text{env}}$', fontsize = 4)
        ax_T.legend(fontsize=4, loc='upper left')
        ax_PI.legend(fontsize=4, loc='upper right')

        os.makedirs(outputpath, exist_ok=True)
        out_file = outputpath + f'plot_PI_large_times.pdf'
        plt.savefig(out_file, dpi = 500, transparent=True, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_file}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Paper figures, Sec. 4.2 (non-linear response) and Sec. 4.3 (long times)
    os.makedirs("./paper_figures/", exist_ok=True)
    plot_MIS_rBRSSS_temperature()          # fig:nonlin_time    plot_MIS_rBRSSS.pdf
    plot_MIS_rBRSSS_PI(method="MIS")       # fig:nonlin_ps (a)  plot_nonlin_ps_MIS.pdf
    plot_MIS_rBRSSS_PI(method="lambda")    # fig:nonlin_ps (b)  plot_nonlin_ps_rBRSSS.pdf
    plot_Large_Scan_rBRSSS()               # fig:nonlin_scan    plot_Large_Scan_Non_Linear.pdf
    plot_large_times_PI(A=0.1)             # fig:LargeTimesEvol plot_PI_large_times.pdf
    plot_large_times(A=0.1)                # fig:LargeTimes     plot_RTA_large_times.pdf
