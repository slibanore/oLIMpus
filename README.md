<p align="center">
  <img src="oLIMpus/logo.jpeg" alt="oLIMpus Logo" width="300"/>
</p>

# oLIMpus: An Effective Model for Line Intensity Mapping Auto- and Cross-Power Spectra in Cosmic Dawn and Reionization

---

`oLIMpus` is an actively maintained Python framework for modelling line intensity mapping (LIM) signals during cosmic dawn and the epoch of reionization. It computes **non-linear auto- and cross-power spectra** of star-forming lines analytically, in milliseconds, and generates **coeval boxes** and **lightcones** from the same model.

The 21-cm signal comes from [`Zeus21`](https://github.com/ZeusCosmo/Zeus21). **In v2 this is an ordinary dependency, not a vendored copy** — see [Installation](#️-installation).

---

<p align="center">
  <img src="oLIMpus/flowchart.png" alt="oLIMpus Flowchart" width="100%"/>
</p>

---

## 🆕 What changed in v2

| | v1 | v2 |
|---|---|---|
| Zeus21 | vendored under `oLIMpus/zeus21_local/` | ordinary dependency, branch `zeus21_hack` |
| `LIM_modeling.py` | — | renamed **`coefficients_LIM.py`** |
| `LIM_luminosities.py` | — | renamed **`luminosities_LIM.py`** |
| burstiness | notebook patches | **`burstiness_LIM.py`**, in the package |
| version number | hard-coded in `setup.py` | derived from `VERSION` + git history |

---

## ⚙️ Installation

We recommend a fresh environment based on **python 3.10** or newer.

If you do not already have [`CLASS`](https://github.com/lesgourg/class_public/), install it first (adapting the Makefile to your `gcc`):

```bash
git clone https://github.com/lesgourg/class_public.git class
cd class/
make
cd python/
pip install .
```

Then, from the directory where you cloned oLIMpus:

```bash
pip install .
```

This pulls in `Zeus21` from the **`zeus21_hack`** branch automatically — it is declared in `install_requires` as a PEP 508 direct reference. oLIMpus v2 will not work against `main`.

For development, install in editable mode so the version number tracks your commits:

```bash
pip install -e .
```

If you use **conda**:

```bash
chmod 755 setup_env.sh
./setup_env.sh
```

which creates the `oLIMpus` environment, installs the code and its dependencies, and adds a jupyter kernel for the tutorials.

`make init`, `make install`, `make dev`, `make version` and `make clean` wrap the same commands.

---

## 🔧 Modules

- **`inputs_LIM.py`**
  `Line_Parameters`: the line and its model, the smoothing radius `R0`, the observable (`Inu` or `Tnu`), shot noise, the lognormal order, luminosity scatter, and the burstiness parameters.

- **`luminosities_LIM.py`**
  `L(Mh, z)` or `L(SFR, z)` models, dispatched by name: `Yang24`, `THESAN21`, `Lagache18`, `Li16`, `Yang21`, `COMAP_fiducial`, `powerlaw_SFR`, `JWST_calibrated`. Lines: OIII (5007, 4960, 4364), OII, Ha, Hb, CII, CO(1-0), CO(2-1).

- **`coefficients_LIM.py`**
  `get_LIM_coefficients`: `sigma_R0(z)`, the EPS conditional HMF, the lognormal coefficients `gamma_R` and `gamma_R^NL` in both Lagrangian and Eulerian space, `phi_LtoE`, the mean intensity `Inu_bar(z)` and the shot noise.

- **`burstiness_LIM.py`** *(new)*
  The Ornstein–Uhlenbeck burstiness model of [arXiv:2605.13967](https://arxiv.org/abs/2605.13967): `V_lambda` (Eq. 5) and `V_12` (Eqs. 8–9) in closed form, the shot-noise boost `1 + V_lambda`, and the cross-line coefficient `R_12` (Eq. 11). Switched on with `Line_Parameters(BURSTY_FLAG=True)`.

- **`correlations_LIM.py`**
  `Power_Spectra_LIM`: the two-point function of two lognormals, the Hankel transform to `P_nu(k, z)`, Kaiser RSD, Fingers-of-God, the shot-noise window, and line-line cross spectra.

- **`maps_LIM.py`**
  `CoevalBox_LIM_analytical` (fast, lognormal), `CoevalBox_percell` (the slow cell-by-cell benchmark), `generate_asym_boxes` for non-cubic geometries, and `build_lightcone`.

- **`analysis.py`**
  `run_oLIMpus`: a convenience wrapper that runs the line and 21-cm calculations together. It rebuilds CLASS on every call — for parameter scans, build the cosmology once and call the classes directly, as the tutorials do.

---

## 📚 Tutorials

In `Tutorials/`, in order:

- **`#1: oLIMpus.ipynb`** — the LIM and 21-cm auto- and cross-power spectra, and how the pieces fit together
- **`#2: boxes_and_lightcones.ipynb`** — coeval boxes, the cell-by-cell benchmark, lightcones
- **`#3: explore_parameters.ipynb`** — how the power spectrum responds to the astrophysical and line parameters
- **`#4: EoR_correlation.ipynb`** — the line × 21-cm correlation through cosmic dawn and reionization
- **`#5: burstiness.ipynb`** — the burstiness model and the shot-noise boost

---

## 📄 Relevant Publications

- Libanore, Mu&ntilde;oz and Kovetz, *oLIMpus: An Effective Model for Line Intensity Mapping Auto- and Cross-Power Spectra in Cosmic Dawn and Reionization*, [arXiv:2507.15922](https://arxiv.org/abs/2507.15922)

- Kovetz, Lazare, Libanore, Mu&ntilde;oz, Vanzan, *When galaxies burst: enhanced shot-noise for line-intensity mapping in the JWST era*, [arXiv:2605.13967](https://arxiv.org/abs/2605.13967)

- Libanore, Kovetz, Mu&ntilde;oz, Sklansky, Th&eacute;lie, *A New Boundary Condition on Reionization*, [arXiv:2509.08886](https://arxiv.org/abs/2509.08886)

- Sklansky et al., *In preparation*

- Mu&ntilde;oz, *An Effective Model for the Cosmic-Dawn 21-cm Signal*, [arXiv:2302.08506](https://arxiv.org/abs/2302.08506)

- Cruz, Mu&ntilde;oz, Sabti and Kamionkowski, *The First Billion Years in Seconds: An Effective Model for the 21-cm Signal with Population III Stars*, [arXiv:2407.18294](https://arxiv.org/abs/2407.18294)

---

## 📬 Contact

**Sarah Libanore**
📧 [libanore@bgu.ac.il](mailto:libanore@bgu.ac.il)

---

> ⭐ If you use **oLIMpus** in your work, please cite the relevant papers!
