# Hydrodynamic attractors in periodically driven systems — figure scripts

Data and plotting scripts that reproduce the figures of

> A. Mazeliauskas, T. Mitra, S. Schlichting, S. Schneider, L. N. Onwuka and M. Vrdoljak,
> *A systematic study of hydrodynamic attractors in periodically driven systems at weak and strong couplings*,
> arXiv:XXXX.XXXXX

The system is a conformal fluid in a homogeneous, periodically driven metric with scale factor
a(t) = 1 + A sin(ωt). We compare Navier-Stokes, MIS and rBRSSS hydrodynamics with RTA, Yang-Mills (EKT) and
scalar φ⁴ kinetic theory and with holography. The hydrodynamic equations are solved by the scripts
themselves. Kinetic theory and holography results are read from the precomputed files in `data/`.

## Quick start

```bash
pip install numpy scipy matplotlib SciencePlots
# run from the repository root (the scripts use relative paths)
python3 plot_small_amplitude.py               # Figs. 1, 4-7
python3 plot_small_amplitude_holography.py    # Figs. 8-10
python3 plot_nonlinear_long_time_RTA.py       # Figs. 11-15
python3 plot_holography_QNM.py                # Figs. 2-3
```

The figures are written to `paper_figures/`, with the same file names that `\includegraphics` uses in the
paper source.

Requirements:
- Python 3 with `numpy`, `scipy` and `matplotlib`. Tested with Python 3.14, numpy 2.5, scipy 1.18 and
  matplotlib 3.11.
- [SciencePlots](https://github.com/garrettj403/SciencePlots) (MIT licence), which provides the `science` +
  `nature` matplotlib styles used for all figures. Install it with
  ```bash
  pip install SciencePlots
  # or: conda install -c conda-forge scienceplots
  ```
  SciencePlots is optional: if it is missing or fails to load, the scripts print a warning and fall back to
  the default matplotlib style, so the figures will not look like the published ones.
- With SciencePlots only: a LaTeX installation, because the `science` style renders all text with LaTeX. On Debian/Ubuntu:
  `sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super dvipng`.
- The four scripts together take about 1.5 minutes. Most of that is the long-time evolution in Figs. 14-15,
  which is integrated on 10⁷ time points.

## Figures

Figure numbers follow the current version of the paper.

| Fig. | Output file(s) in `paper_figures/` | Content | Script → function | Input data |
|---|---|---|---|---|
| 1 | `plot_rate.pdf` | Expansion rate ȧ/(aAω) for A = 0.1, 0.5, 0.8, 0.9 | `plot_small_amplitude.py` → `plot_expansion_rate` | — (analytic) |
| 2 | `GR_vs_omega_tau_pi.pdf` | Re/Im of the retarded correlator vs ωτπ, holography vs hydrodynamics | `plot_holography_QNM.py` → `plot_GR_vs_omega_tau_pi` | `data/holography/QNM/` |
| 3 | `QNM.pdf`, `GR_residue_2vs4_poles.pdf` | Holographic QNM spectrum vs MIS mode (left); frequency-weighted residues (right) | `plot_holography_QNM.py` → `plot_QNM_spectrum`, `plot_residue_2vs4_poles` | `data/holography/QNM/` |
| 4 | `plot_MIS_T_A0.1.pdf` | T/T₀ and π/(P_eq 𝒩) vs time: NS, MIS, linear MIS, free streaming; A = 0.1 | `plot_small_amplitude.py` → `plot_MIS_temperature_evolution` | — (ODEs) |
| 5 | `plot_phase_space_A0.1.pdf` | π vs π_NS: NS, MIS, linear MIS, free streaming; A = 0.1 | `plot_small_amplitude.py` → `plot_phase_space_comparison` | — (ODEs) |
| 6 | `plot_kinetic_T_A0.1.pdf` | T/T₀ and π/(P_eq 𝒩) vs time: RTA, Yang-Mills, scalar, MIS; A = 0.1 | `plot_small_amplitude.py` → `plot_kinetic_temperature_evolution` | `data/RTA/small_amplitude/`, `data/YangMills/small_amplitude/`, `data/scalar_phi4/small_amplitude/` |
| 7 | `plot_kinetic_phase_space_A0.1.pdf` | π vs π_NS for the same theories | `plot_small_amplitude.py` → `plot_kinetic_phase_space_comparison` | same as Fig. 6 |
| 8 | `plot_holo_T_A0.001.pdf` | T/T₀ and π/(P_eq 𝒩) vs time: holography, NS, MIS (κ ≠ 0 and κ = 0); A = 0.001 | `plot_small_amplitude_holography.py` → `plot_holo_temperature_evolution` | `data/holography/small_amplitude/` |
| 9 | `plot_holo_T_NSsub_A0.001.pdf` | (π − π_NS)/(P_eq 𝒩) at ωτπ = 0.1 | `plot_small_amplitude_holography.py` → `plot_holo_temperature_evolution_NSsub` | `data/holography/small_amplitude/` |
| 10 | `plot_holo_phase_space_A0.001.pdf` | π vs π_NS: holography and MIS | `plot_small_amplitude_holography.py` → `plot_holo_phase_space_comparison` | `data/holography/small_amplitude/` |
| 11 | `plot_MIS_rBRSSS.pdf` | Non-linear regime A = 0.3, 0.7: T/T₀ and π/P_eq vs time; RTA, NS, MIS, rBRSSS | `plot_nonlinear_long_time_RTA.py` → `plot_MIS_rBRSSS_temperature` | `data/RTA/nonlinear/` |
| 12a | `plot_nonlin_ps_MIS.pdf` | Non-linear π vs π_NS: RTA, MIS, NS | `plot_nonlinear_long_time_RTA.py` → `plot_MIS_rBRSSS_PI(method="MIS")` | `data/RTA/nonlinear/` |
| 12b | `plot_nonlin_ps_rBRSSS.pdf` | Non-linear π vs π_NS: RTA, rBRSSS, NS | `plot_nonlinear_long_time_RTA.py` → `plot_MIS_rBRSSS_PI(method="lambda")` | `data/RTA/nonlinear/` |
| 13 | `plot_Large_Scan_Non_Linear.pdf` | Scan over A = 0.1…0.9 and Kn_max = 0.05…0.5: RTA, rBRSSS, NS | `plot_nonlinear_long_time_RTA.py` → `plot_Large_Scan_rBRSSS` | `data/RTA/nonlinear/` |
| 14 | `plot_PI_large_times.pdf` | Long-time evolution, A = 0.1, ωτπ = 3.2: temperature and shear-stress envelopes | `plot_nonlinear_long_time_RTA.py` → `plot_large_times_PI` | `data/RTA/long_time/` |
| 15 | `plot_RTA_large_times.pdf` | Long-time RTA trajectory π vs π_NS, coloured by ωt | `plot_nonlinear_long_time_RTA.py` → `plot_large_times` | `data/RTA/long_time/` |

To make a single figure, import the script and call its function, e.g.
`python3 -c "import plot_small_amplitude as p; p.plot_kinetic_temperature_evolution(A=0.1)"`.

## Repository layout

```
├── plot_small_amplitude.py              # Figs. 1, 4-7   (hydro + kinetic theory, A = 0.1)
├── plot_small_amplitude_holography.py   # Figs. 8-10    (holography, A = 0.001)
├── plot_nonlinear_long_time_RTA.py      # Figs. 11-15   (non-linear and long-time RTA)
├── plot_holography_QNM.py               # Figs. 2-3     (holographic G_R, QNMs, residues)
├── hydro_boltzmann.py      # NS / MIS / rBRSSS ODEs and free streaming; massless Boltzmann EoS, τπ = 5 η/(e+P)
├── hydro_holography.py     # NS / MIS ODEs with N=4 SYM transport coefficients, τπ = (2 − ln 2)/(πT), κ term
├── data/
│   ├── RTA/
│   │   ├── small_amplitude/     # Figs. 6-7
│   │   ├── nonlinear/           # Figs. 11-13
│   │   └── long_time/           # Figs. 14-15
│   ├── YangMills/small_amplitude/      # Figs. 6-7
│   ├── scalar_phi4/small_amplitude/    # Figs. 6-7
│   └── holography/
│       ├── small_amplitude/     # Figs. 8-10
│       └── QNM/                 # Figs. 2-3
├── paper_figures/           # output (generated)
└── LICENSE                  # CC BY 4.0
```

## Data files

All runs start in equilibrium at T₀ (T₀ = 1 except in holography, where T₀ = 1/π) with a(t) = 1 + A sin(ωt).
`wtaupi` in a file name is the dimensionless frequency ωτπ at T₀. `omega` is ω in units of T₀.

### `data/RTA/` — RTA kinetic theory (τπ = τ_R = 5(η/s)/T)

| Folder | File name pattern | Parameters | Columns |
|---|---|---|---|
| `small_amplitude/` | `RTA_etas0.56_A0.1_wtaupi{0.1,0.5,2.0,10.0}.txt` | η/s = 0.56, A = 0.1. Evolved to ωt = 10π. | `t  e  P_T  P_L  ȧ/a  T  e_NS  P_L,NS  T_NS  a  ȧ  ωt` |
| `nonlinear/` | `RTA_etas0.08_A{A}_omega{ω}.txt` | η/s = 0.08 (τπ = 0.4), A = 0.1, 0.3, 0.5, 0.7, 0.9, four frequencies per A. Evolved to ωt = 100.1. | same 12 columns |
| `long_time/` | `RTA_etas0.08_A0.1_omega8.0.txt` | η/s = 0.08, A = 0.1, ω = 8 (ωτπ = 3.2). Evolved to ωt = 10⁴. | first 11 columns (no `ωt`) |

Numerical parameters of all RTA runs: moments up to `lmax = 64`, base step ω·dt = 8·10⁻⁴.
The shear stress is π = (2/3)(P_L − P_T).

### `data/YangMills/small_amplitude/` — Yang-Mills effective kinetic theory

`YM_lambda10_taupi5.2_A0.1_wtaupi{0.1,0.5,2.0,10}.out`: gluons with N_c = 3 at 't Hooft coupling λ = 10.
The drive frequency ω was chosen from ωτπ using τπ = 5.2(η/s)/T₀ with η/s = 0.56 (the `taupi5.2` tag).
Columns, labelled in the file header: 1 `time`, 2 `lambda`, 3 `Meff2`, 4 `Tstar`, 5 `n`, 6 `ff`, 7 `pff`,
8 `entropy`, 9–18 the components `T00 … Tzz` of T^μν. The scripts use e = T00 and π = Tzz − e/3.

### `data/scalar_phi4/small_amplitude/` — scalar φ⁴ kinetic theory

`scalar_A0.1_omega{ω:.6f}_wtaupi{0.1,0.5,2.0,10.0}_ntheta256.txt`: τπ = 6.10517(η/s)/T₀ with η/s = 0.56.
The run uses 256 angular grid points. The file name gives ω with 6 decimals so that each run is identified
exactly.
Columns, as in the file header: `s  t  a  adot  exprate  tstar  mdsqr  n  e  pl  T  mu`. The scripts use π/P = P_L/(e/3) − 1.

### `data/holography/` — N = 4 SYM (η/s = 1/4π, T₀ = 1/π, τπ = (2 − ln 2)/(πT))

| Folder | Files | Content |
|---|---|---|
| `small_amplitude/` | `holography_A0.001_wtaupi{0.1,0.3,0.5,1}.dat` | Linear response at A = 0.001. Columns `t  π_NS/(ωτπ P_eq)  π/(ωτπ P_eq)  T` |
| `QNM/` | `GR_holography_wtaupi0to1.txt` | τπ G_R/(4η/3) for x = ωτπ ∈ [0, 1]. Columns `x  Re  Im`. Used for Fig. 2 |
| `QNM/` | `QNM.txt` | The 10 lowest-lying QNM frequencies 𝔴ₙ = Ωₙ/(2πT), one complex number per line (Mathematica syntax). Used for Fig. 3 |
| `QNM/` | `QNM_residues.txt` | Residue prefactors cₙ (T = 1) of the QNM terms cₙ x/(x²/τπ² − Ωₙ²), sorted by \|Ωₙ\|. Columns `Re Ωₙ  Im Ωₙ  Re cₙ  Im cₙ`. Used for Fig. 3 |

`GR_holography_wtaupi0to1.txt` and `QNM_residues.txt` were evaluated (the residues with 50-digit `mpmath`
arithmetic) from Mathematica expressions for G_R and the QNM residues, which are not included here.

## Citation

If you use these data or scripts, please cite the paper above.

```bibtex
@article{Mazeliauskas:XXXX,
  author  = {Mazeliauskas, Aleksas and Mitra, Toshali and Schlichting, S{\"o}ren and Schneider, Simon and Onwuka, Louis Nwakile and Vrdoljak, Martin},
  title   = {{A systematic study of hydrodynamic attractors in periodically driven systems at weak and strong couplings}},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  year    = {2026}
}
```

## License

The scripts and data in this repository are released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
licence; see [`LICENSE`](LICENSE). You may share and adapt them for any purpose, provided you give
appropriate credit by citing the paper above.
