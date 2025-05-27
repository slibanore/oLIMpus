"""
Takes inputs for LIM and stores them in useful classes

Author: Sarah Libanore
BGU - April 2025
"""

import astropy.constants as cu
import astropy.units as u
import numpy as np 

class LineParams_Input:
    def __init__ (self, 
                 LINE = 'OIII', # which line
                 LINE_MODEL = 'SFRD', # model of the line luminosity
                 OBSERVABLE_LIM = 'Inu', # observe intensity in Jy/sr or mK
                 _R = 0.5, # resolution for smoothing
                 sigma_LMh = 0., # stochasticity in the L-Mh relation
                 shot_noise = False, # add shot noise to the LIM power spectrum

                 quadratic_lognormal = False, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
                 ):
        
        self.LINE = LINE
        self.OBSERVABLE_LIM = OBSERVABLE_LIM
        self._R = _R
        self.LINE_MODEL = LINE_MODEL
        self.sigma_LMh = sigma_LMh
        self.shot_noise = shot_noise

        # !!! move to User_Params
        self.quadratic_lognormal = quadratic_lognormal



class Line_Parameters:

    "Class to pass the parameters of LIM as input"

    def __init__(self, LineParams_Input, User_Parameters):

        self.LINE = LineParams_Input.LINE # which line to use 
        
        if self.LINE == 'OIII':
            lambda_line = 4960*u.AA 
        elif self.LINE == 'Ha':
            lambda_line = 6563*u.AA
        elif self.LINE == 'Hb':
            lambda_line = 4861*u.AA 
        elif self.LINE == 'CII':
            lambda_line = 1.58e6*u.AA 
        elif self.LINE == 'CO21': # 2-1 transition
            lambda_line = 1.3e7*u.AA 
        elif self.LINE == 'CO10': # 2-1 transition
            lambda_line = 2.6e7*u.AA 
        self.nu_rest = (cu.c / (lambda_line)).to(u.Hz) # rest frame frequency in Hz 

        self.OBSERVABLE_LIM = LineParams_Input.OBSERVABLE_LIM

        # resolution in Mpc, cannot go below the Rmin defined in Cosmo_Params               
        if LineParams_Input._R < User_Parameters.MIN_R_NONLINEAR:
            print('Too small R, we  use instead MIN_R_NONLINEAR')
            self._R = User_Parameters.MIN_R_NONLINEAR
        else:
            self._R = LineParams_Input._R 

        self.LINE_MODEL = LineParams_Input.LINE_MODEL 
        try:
            self.sigma_LMh = LineParams_Input.sigma_LMh.value*np.log(10) if LineParams_Input.sigma_LMh.unit == u.dex else LineParams_Input.sigma_LMh
        except:
            self.sigma_LMh = LineParams_Input.sigma_LMh
        self.shot_noise = LineParams_Input.shot_noise

        # !!! move to User_Params
        self.quadratic_lognormal = LineParams_Input.quadratic_lognormal


####################
# Define parameters for some of the models included in the LIM file

Yang24_OIII_params = {
        'alpha': 9.82e-2,
        'beta': 6.90e-1,
        'N': 2.75e7,
        'SFR1': 1.24e2,
        }

Yang24_OII_params = {
        'alpha': -2.43e-1,
        'beta': 2.5,
        'N': 2.14e6,
        'SFR1': 5.91e1,
        }

Yang24_Ha_params = {
        'alpha': 9.94e-3,
        'beta': 5.25e-1,
        'N': 4.54e7,
        'SFR1': 3.18e1,
        }

Yang24_Hb_params = {
        'alpha': 7.98e-3,
        'beta': 5.61e-1,
        'N': 1.61e7,
        'SFR1': 1.74e1,
        }


THESAN21_OIII_params = {'a': 7.84,
    'ma': 1.24,
    'mb': 1.19,
    'log10_SFR_b': 0.,
    'mc': 0.53, 
    'log10_SFR_c': 0.66, 
    }

THESAN21_OII_params = {'a': 7.08,
    'ma': 1.11,
    'mb': 1.31,
    'log10_SFR_b': 0.,
    'mc': 0.64, 
    'log10_SFR_c': 0.54, 
    }

THESAN21_Ha_params = {'a': 8.08,
    'ma': 0.96,
    'mb': 0.88,
    'log10_SFR_b': 0.,
    'mc': 0.45, 
    'log10_SFR_c': 0.96, 
    }

THESAN21_Hb_params = {'a': 7.62,
    'ma': 0.96,
    'mb': 0.86,
    'log10_SFR_b': 0.,
    'mc': 0.41, 
    'log10_SFR_c': 0.96, 
    }

Lagache18_CII_params = {'alpha_SFR_0': 1.4-0.07*10,
        'beta_SFR_0': 7.1-0.07*10,
        'alpha_SFR': 0.,
        'beta_SFR': 0.,
        }

Yang21_CO21_params = {
    'A':1.
}

Li16_C021_params = {
    'alpha':1.11,
    'beta':0.6,
    'dMF':1.,
    'L0':4.9e-5,
    'sigma_SFR':0.3*u.dex
}