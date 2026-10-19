"""
Figs. 2-3: holographic retarded correlator, QNM spectrum and QNM residues.

N = 4 SYM with T = 1, input in data/holography/QNM/:

  GR_holography_wtaupi0to1.txt  tau_pi G_R(x)/(4 eta/3), x = omega tau_pi in [0, 1]
  QNM.txt                       lowest-lying QNM frequencies w_n = Omega_n/(2 pi T)
  QNM_residues.txt              residue prefactors c_n of the QNM terms c_n x/(x^2/tau_pi^2 - Omega_n^2)

The two tables were evaluated from Mathematica expressions of G_R and of the QNM residues.
"""
import math
import os

import numpy as np
import matplotlib.pyplot as plt

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

# same colours as in the other plot scripts
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
COLORS = colors
# colour and line style per theory, as in Figs. 4-10
THEORY = {
    'Holography':     dict(color=COLORS[7], ls='--'),
    'MIS':            dict(color=COLORS[1], ls='-'),
    'free-streaming': dict(color=COLORS[2], ls='--'),
    'RTA':            dict(color=COLORS[5], ls=':'),
}

# Global Plot Settings
FRAME_WIDTH = 2.5

# ==========================================
# Parameters
# ==========================================

DATA_DIR = "./data/holography/QNM/"
FILE_QNM = DATA_DIR + "QNM.txt"
TABLE_GR = DATA_DIR + "GR_holography_wtaupi0to1.txt"
TABLE_RESIDUES = DATA_DIR + "QNM_residues.txt"

TAU_PI = (2 - math.log(2)) / (2 * math.pi)   # holographic tau_pi at T = 1
N_POINTS = 500

# ==========================================
# Data
# ==========================================


def load_qnm(file_qnm=FILE_QNM):
    """QNM frequencies w_n = Omega_n/(2 pi T), one complex number per line in Mathematica syntax."""
    with open(file_qnm) as f:
        return np.array([eval(line.strip(), {"I": 1j, "__builtins__": {}}) for line in f if line.strip()])

# ==========================================
# Plotting Functions
# ==========================================


def plot_GR_vs_omega_tau_pi(outputpath="./paper_figures/"):
    """Fig. 2: Re/Im of tau_pi [2 G^{eta eta,eta eta} - (2/3) G^{tt,eta eta}]/(4 eta/3) for holography, RTA, free streaming and MIS."""
    scale = 0.8
    xs, re_gr, im_gr = np.loadtxt(TABLE_GR, unpack=True)

    # RTA: (i x + (Ck/Ct) x^2)/(1 - i x) with (Ck, Ct) = (0, 5); free streaming: -Ct/5
    Ck, Ct = 0, 5
    # (style key, legend label, curve), in plotting/legend order
    curves = [
        ('RTA', r'RTA, MIS $\kappa=0$', (1j * xs + (Ck / Ct) * xs**2) / (1 - 1j * xs)),
        ('Holography', 'Holography', re_gr + 1j * im_gr),
        ('MIS', r'MIS $\kappa\neq0$',
         (1j * xs + xs**2 / (2 * np.pi) / TAU_PI) / (1 - 1j * xs)),   # kappa/(2 eta) = 1/(2 pi T)
        ('free-streaming', 'free-streaming', np.full_like(xs, -Ct / 5, dtype=complex)),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(scale * 2.25 * FRAME_WIDTH, scale * FRAME_WIDTH))
    for ax, part, name in zip(axs, (np.real, np.imag), ("Re", "Im")):
        ax.hlines(0, xs[0], xs[-1], 'k', lw=0.5)
        for key, label, G in curves:
            ax.plot(xs, part(G), **THEORY[key], label=label)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xlabel(r'$\omega_0\tau_\pi$')
        ax.set_ylabel(r'$\tau_\pi\,\mathrm{' + name + r'}[2G^{\eta\eta,\eta\eta}-\frac{2}{3}G^{tt,\eta\eta}]/(4\eta/3)$')
    axs[1].legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    out_file = outputpath + "GR_vs_omega_tau_pi.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_QNM_spectrum(outputpath="./paper_figures/"):
    """Fig. 3 (left): holographic QNMs and the MIS pole in w = omega/(2 pi T)."""
    scale = 0.8
    w_n = load_qnm()
    w_MIS = (-1j / TAU_PI) / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(scale * 1.25 * FRAME_WIDTH, scale * FRAME_WIDTH))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.plot(w_n.real, w_n.imag, 'o', color=THEORY['Holography']['color'], ms=4, label='Holography')
    ax.plot(w_MIS.real, w_MIS.imag, 'x', color=THEORY['MIS']['color'], ms=6, mew=1.5,
            label='MIS')
    ax.set_xlabel(r'$\mathrm{Re}\,\mathfrak{w}$')
    ax.set_ylabel(r'$\mathrm{Im}\,\mathfrak{w}$')
    ax.set_ylim(None, 2.7)   # room for the legend
    ax.set_yticks([-4, -2, 0, 2])
    ax.legend(fontsize=7, loc='upper right', frameon=False)

    fig.tight_layout()
    out_file = outputpath + "QNM.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_residue_2vs4_poles(x_range=(0.0, 20.0), outputpath="./paper_figures/"):
    """Fig. 3 (right): Re of the residue sum for the lowest 1 and 2 QNM pairs, and the MIS residue."""
    scale = 0.8
    xs = np.linspace(*x_range, N_POINTS)

    re_Om, im_Om, re_c, im_c = np.loadtxt(TABLE_RESIDUES, unpack=True, ndmin=2)
    Omega_n, c_n = re_Om + 1j * im_Om, re_c + 1j * im_c   # sorted by |Omega_n|
    terms = [c * xs / (xs**2 / TAU_PI**2 - Om**2) for Om, c in zip(Omega_n, c_n)]
    re_2poles = np.sum(terms[:2], axis=0).real
    re_4poles = np.sum(terms[:4], axis=0).real

    mis_residue = -(xs * (1 - 1 / (2 - math.log(2)))) / (1 + xs**2)

    fig, ax = plt.subplots(figsize=(scale * 1.25 * FRAME_WIDTH, scale * FRAME_WIDTH))
    ax.hlines(0, *x_range, 'k', lw=0.5)
    ax.plot(xs, re_2poles, color=COLORS[3], ls='-.', label='Holography, 2 poles')
    ax.plot(xs, re_4poles, **THEORY['Holography'], label='Holography, 4 poles')
    ax.plot(xs, mis_residue, **THEORY['MIS'], label='MIS')
    ax.set_xlim(*x_range)
    ax.set_ylim(-10, 230)   # room for the legend and the inset
    ax.set_xlabel(r'$\omega_0\tau_\pi$')
    ax.set_ylabel(r'$\mathrm{Re}[\tau_\pi\gamma_n/(4\eta/3)]$')
    ax.legend(fontsize=7, loc='upper right', frameon=False)

    # the MIS residue is tiny on the main axes: inset with its own range
    ax_in = ax.inset_axes([0.6, 0.36, 0.36, 0.24])
    ax_in.hlines(0, *x_range, 'k', lw=0.5)
    ax_in.plot(xs, mis_residue, **THEORY['MIS'])
    ax_in.set_xlim(*x_range)
    ax_in.tick_params(labelsize=7)

    fig.tight_layout()
    out_file = outputpath + "GR_residue_2vs4_poles.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_file}")

# ==========================================
# Main Execution
# ==========================================


if __name__ == "__main__":
    # Paper figures, Sec. 3 (holographic correlator and QNMs)
    os.makedirs("./paper_figures/", exist_ok=True)
    plot_GR_vs_omega_tau_pi()       # Fig:Compare_HBRsss   GR_vs_omega_tau_pi.pdf
    plot_QNM_spectrum()             # Fig:QNM spectrum     QNM.pdf
    plot_residue_2vs4_poles()       # Fig:QNM spectrum     GR_residue_2vs4_poles.pdf
