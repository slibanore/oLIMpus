<p align="center">
  <img src="oLIMpus/logo.jpeg" alt="oLIMpus Logo" width="200"/>
</p>

# oLIMpus: An Effective Model for Line Intensity Mapping Auto- and Cross-Power Spectra in the Epoch of Reionization

---

**oLIMpus** is an actively maintained and expanding Python-based framework for simulating line intensity mapping signals during the Epoch of Reionization (EoR). It provides a fast and efficient way to compute **non-linear power spectra** of star-forming lines, and generates both **coeval boxes** and **lightcones**.

The 21cm signal is introduced by interfacing oLIMpus with [`Zeus21`](https://github.com/JulianBMunoz/Zeus21), a public code for 21cm signal modeling at cosmic dawn. A version of `Zeus21` is included in this repository as a submodule. Note that the original `Zeus21` currently models only the power spectrum during cosmic dawn; support for the reionization era is under active development.

---

<p align="center">
  <img src="oLIMpus/flowchart.jpeg" alt="oLIMpus Flowchart" width="600"/>
</p>

---

## 🔧 Included Modules

- **`olim_power.py`**  
  Computes auto- and cross-power spectra of various emission lines, including astrophysical nonlinearities and redshift-space distortions.

- **`olim_lightcone.py`**  
  Constructs lightcone cubes for emission lines across redshift, including interpolation and proper line-of-sight evolution.

- **`olim_box.py`**  
  Generates coeval 3D boxes of intensity and density fields at a fixed redshift.

- **`zeus21/`** (submodule)  
  Modified version of [`zeus21`](https://github.com/zeus21/zeus21) used to compute 21cm power spectra and fields during cosmic dawn (reionization modeling in progress).

- **`params/`**  
  Contains example configuration files with cosmological and astrophysical parameters.

---

## 📚 Tutorials

The following notebooks and scripts will help you get started:

- **`tutorial_1_power_spectrum.ipynb`** – Compute and visualize the LIM power spectra at given redshifts  
- **`tutorial_2_lightcone.ipynb`** – Generate a lightcone cube for an emission line  
- **`tutorial_3_cross_21cm.ipynb`** – Cross-correlate LIM signals with 21cm from `zeus21`  
- **`tutorial_4_custom_params.ipynb`** – Use custom astrophysical and cosmological parameters

> 📌 You can find all tutorials in the `tutorials/` folder.

---

## 📄 Relevant Publications

- Libanore, Mu&ntilde;oz and Kovetz, *oLIMpus: An Effective Model for Line Intensity Mapping Auto- and Cross-Power Spectra in the Epoch of Reionization*, [arXiv:2506.YYYYY](https://arxiv.org/abs/2506.YYYYY)

- Mu&ntilde;oz, *An Effective Model for the Cosmic-Dawn 21-cm Signal*, [arXiv:2302.08506](https://arxiv.org/abs/2302.08506)

- Sklanksy et al., *In preparation*

- Cruz, Mu&ntilde;oz, Sabti and Kamionkowski, *The First Billion Years in Seconds: An Effective Model for the 21-cm Signal with Population III Stars*, [arXiv:2407.18294](https://arxiv.org/abs/2407.18294)

---

## 📬 Contact

For questions, suggestions, or help using the code, please contact:

**Sarah Libanore**  
📧 [libanore@bgu.ac.il](mailto:libanore@bgu.ac.il)  
Ben Gurion University of the Negev

---

> ⭐ If you use **oLIMpus** in your research, please cite the relevant papers!
