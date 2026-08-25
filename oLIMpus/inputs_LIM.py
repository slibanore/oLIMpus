"""
Takes inputs for LIM and stores them in useful classes

Author: Sarah Libanore
BGU - June 2026
"""


# all the imports you will need throughtout the run 
from zeus21 import constants, cosmology, sfrd, z21_utilities, reionization, inputs
import astropy.constants as cu
import astropy.units as u
import numpy as np 
from scipy.integrate import simpson
from scipy.stats import lognorm
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import mcfit
import numexpr as ne
import powerbox as pbox
from tqdm import trange, tqdm
import pickle
import os
from matplotlib import colors as cc 
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt 

from dataclasses import dataclass, field as _field, InitVar

@dataclass(kw_only=True)
class Line_Parameters:
    """
    Parameters associated with star-forming lines and oLIMpus implementation.

    Parameters
    ----------
    LINE: str 
        Which line to compute. Available choices: "OIII5007" (same as "OIII"), "OIII4960", "OIII4364", "OII", "Ha", "Hb", "CII", "CO21", "CO10", "SFRD"        
    LINE_MODEL: str 
        How to compute the line luminosity. Can pass the name of any of the functions in luminosities_LIM.py
    OBSERVABLE_LIM: str 
        Whether to compute the line intensity in Jy/sr (if "Inu") or the brightness temperature in uK (if "Tnu")
    _R: float
        Scale below which the astrophysical model cannot be trusted (the local process is computed inside a sphere with this radius). Has to be larger than MIN_R_NONLINEAR in UserParams (see inputs.py in Zeus21)
    shot_noise: bool
        Whether to include shot noise in the LIM power spectrum calculation, default True
    quadratic_rhoL: bool 
        Whether to include the second order in the lognormal expansion of the luminosity density wrt the density field, default True 
    stoch_type: str
        Whether to anchor the stochastic enhancement to the mean ("mean") or on the median ("median")
        TODO: Add burstiness option from 2605.13967
    sigma_LMh_dex: float = 0.
        Deterministic rms normal scatter in dex, CONSTANT in redshift.
    sigma_LMh_dex_of_z: callable or None
        Optional callable z -> scatter in dex, used INSTEAD of
        sigma_LMh_dex wherever the scatter is applied. Supplied when the
        scatter is calibrated as a function of redshift; None (the
        default) reproduces the constant behaviour exactly. It is
        evaluated with whatever z the caller holds, so it must broadcast:
        a scalar z gives a scalar, and the (nz, 1) redshift column used by
        the luminosity-density and shot-noise integrands gives an (nz, 1)
        column.
    line_dict: dict
        Dictionary containing all the quantities required to estimate the line luminosity (see at the bottom of this file) 

    Attributes
    ----------
    lambda_line: float
        Rest-frame wavelenght of the line in Angstrom 
    nu_line: float
        Rest-frame frequency of the line in Hz
    """

    ### Default and init=False parameters
    LINE: str = "OIII5007"
    LINE_MODEL: str = "Yang24"
    OBSERVABLE_LIM: str = "Inu"
    DUST_FLAG: bool = False
    BURSTY_FLAG: bool = False

    _R: float = 0.5
    shot_noise: bool = True
    quadratic_rhoL: bool = True

    stoch_type: str = "mean"
    sigma_LMh_dex: float = 0.
    sigma_LMh_dex_of_z: object = None
    sigPS_piv_bursty: float = 2.
    log10M_piv_bursty: float = 11
    sigPS_cap_bursty: float = 0.25
    dsigPS_dlog10M_bursty: float = -0.5
    tauPS_Myr_bursty: float = 25.

    line_dict: dict = None   
    t_Myr_per_line: float = _field(init=False)
    sigma_LMh: float = _field(init=False)

    def __post_init__(self):
        schema = {
            "LINE": (str, {"OIII5007", "OIII", "OIII4960", "OIII4364", "OII", "Ha", "Hb", "CII", "CO21", "CO10", "SFRD"}),
            "OBSERVABLE_LIM": (str, {"Inu", "Tnu"}),
            "DUST_FLAG": (bool, None),
            "BURSTY_FLAG": (bool, None),
            "shot_noise": (bool, None),
            "quadratic_rhoL": (bool, None),
            "stoch_type": (str, {"mean", "median"}),
        }
        inputs.validate_fields(self, schema)
        
        if self.LINE == 'OIII4960':
            self.lambda_line = 4960*u.AA 
            self.t_Myr_per_line = 7.
        elif self.LINE == 'OIII5007' or self.LINE == 'OIII':
            self.lambda_line = 5007*u.AA 
            self.t_Myr_per_line = 7.
        elif self.LINE == 'OII':
            self.lambda_line = 3727.29*u.AA 
            self.t_Myr_per_line = 7.
        elif self.LINE == 'Ha':
            self.lambda_line = 6563*u.AA
            self.t_Myr_per_line = 7.
        elif self.LINE == 'Hb':
            self.lambda_line = 4861*u.AA 
            self.t_Myr_per_line = 7.

        elif self.LINE == 'CII':
            self.lambda_line = 1.58e6*u.AA 
            self.t_Myr_per_line = 50.

        elif self.LINE == 'CO21': # 2-1 transition
            self.lambda_line = 1.3e7*u.AA 
            self.t_Myr_per_line = 80.
        elif self.LINE == 'CO10': # 1-0 transition
            self.lambda_line = 2.6e7*u.AA 
            self.t_Myr_per_line = 80.

        elif self.LINE == 'SFRD':
            self.lambda_line = 1.*u.AA 
            self.t_Myr_per_line = 100.

        self.nu_rest = (cu.c / (self.lambda_line)).to(u.Hz) 

        self.sigma_LMh = self.sigma_LMh_dex * np.log(10.0)

    def sigma_LMh_at(self, z):
        """Scatter of ln L at fixed halo mass, evaluated AT z.

        Returns the constant sigma_LMh when sigma_LMh_dex_of_z is None, so every
        call site behaves exactly as before. Otherwise the callable is evaluated
        at z and converted from dex to natural log. The conversion lives here,
        once, so a caller cannot forget the ln(10).
        """
        if self.sigma_LMh_dex_of_z is None:
            return self.sigma_LMh
        return np.asarray(self.sigma_LMh_dex_of_z(z), dtype=float) * np.log(10.0)

"""
Define dictionaries containing default parameters for the models in luminosities_LIM.py
"""

########################################################
### UV AND OPTICAL 
########################################################

# Yang24: arXiv:2409.03997v2, table 2
# THESAN21: arXiv:2111.02411, table 2 

# OIII lines
Yang24_OIII5007_params = {
        'N': 7.68e7,
        'SFR1': 9.76e1,
        'alpha': 9.48e-2,
        'beta': 9.28e-1,
        }

Yang24_OIII4960_params = {
        'N': 2.61e7,
        'SFR1': 9.60e1,
        'alpha': 9.46e-2,
        'beta': 9.24e-1,
        }

Yang24_OIII4364_params = {
        'N': 6.78e5,
        'SFR1': 1.09e2,
        'alpha': -9.62e-3,
        'beta': 2.25,
        }

THESAN21_OIII_params = {
    'a': 7.84,
    'ma': 1.24,
    'mb': 1.19,
    'mc': 0.53, 
    'log10_SFR_b': 0.,
    'log10_SFR_c': 0.66, 
    }

# OII line
Yang24_OII_params = {
        'N': 2.00e6,
        'SFR1': 6.28e1,
        'alpha': -2.43e-1,
        'beta': 2.49,
        }

THESAN21_OII_params = {
    'a': 7.08,
    'ma': 1.11,
    'mb': 1.31,
    'mc': 0.64, 
    'log10_SFR_b': 0.,
    'log10_SFR_c': 0.54, 
    }

# Ha line
Yang24_Ha_params = {
        'N': 4.28e7,
        'SFR1': 4.42e1,
        'alpha': -4.03e-3,
        'beta': 5.84e-1,
        }

THESAN21_Ha_params = {
    'a': 8.08,
    'ma': 0.96,
    'mb': 0.88,
    'mc': 0.45, 
    'log10_SFR_b': 0.,
    'log10_SFR_c': 0.96, 
    }

# Hb line
Yang24_Hb_params = {
        'N': 1.63e7,
        'SFR1': 1.79e1,
        'alpha': 1.78e-2,
        'beta': 5.77e-1,
        }

THESAN21_Hb_params = {
    'a': 7.62,
    'ma': 0.96,
    'mb': 0.86,
    'mc': 0.41, 
    'log10_SFR_b': 0.,
    'log10_SFR_c': 0.96, 
    }

########################################################
### INFRARED
########################################################

# Lagache18: 1711.00798 anchored at z=10

# CII line
Lagache18_CII_params ={
    'alpha_SFR_0': 1.4-0.07*10,
    'beta_SFR_0': 7.1-0.07*10,
    'alpha_SFR': 0.,
    'beta_SFR': 0.,
    }

########################################################
### SUB-MM CO
########################################################

# Li16: arXiv:
# Yang21: arXiv:2108.07716, scale up/down the empirical fit

# CO 2-1 transition
Li16_C021_params = {
    'alpha':1.11,
    'beta':0.6,
    'dMF':1.,
    'L0':4.9e-5,
    'sigma_SFR':0.3*u.dex
}

Yang21_CO21_params = {
    'A':1.
}

# CO 1-0 transition
Li16_C010_params = {
    'alpha':1.27,
    'beta':-1.,
    'dMF':1.,
    'L0':4.9e-5,
    'sigma_SFR':0.3*u.dex
}

Yang21_CO10_params = {
    'A':1.
}

# COMAP fiducial 
# generic CO line

COMAP_pessimistic_params = {
    'A': -3.7, 
    'B': 7.0, 
    'C': 11.1, 
    'Ms': 12.5, 
    'sigma': 0.36
}

COMAP_realistic_params = {
    'A': -2.75, 
    'B': 0.05, 
    'C': 10.61, 
    'Ms': 12.3, 
    'sigma': 0.42
}

COMAP_realisticplus_params = {
    'A': -2.85, 
    'B': -0.42, 
    'C': 10.63, 
    'Ms': 12.3, 
    'sigma': 0.42
}

COMAP_optimistic_params = {
    'A': -2.4, 
    'B': -0.5, 
    'C': 10.45, 
    'Ms': 12.21, 
    'sigma': 0.36
}

