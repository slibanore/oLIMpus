"""
Code to compute LIM correlation functions and power spectra. Based on correlations.py 

Author: Sarah Libanore
BGU - April 2025

"""

import mcfit
import numpy as np 
import numexpr as ne
from oLIMpus import constants, cosmology 

class Correlations_LIM:
    "Class that calculates and keeps the correlation functions."

    def __init__(self, Line_Parameters, Cosmo_Parameters, ClassCosmo):

        # k array for the power spectrum, based on the same r array as in the 21cm case
        self._klistCF, _dummy_ = mcfit.xi2P(Cosmo_Parameters._Rtabsmoo, l=0, lowring=True)(0*Cosmo_Parameters._Rtabsmoo, extrap=False)
        self.NkCF = len(self._klistCF)

        # linear matter power spectrum at z = 0 in 1/Mpc^3
        self._PklinCF = np.zeros(self.NkCF) 
        for ik, kk in enumerate(self._klistCF):
            self._PklinCF[ik] = ClassCosmo.pk(kk, 0.0) 

        # define function to transform the power specrum into correlation function
        self._xif = mcfit.P2xi(self._klistCF, l=0, lowring=True)

        # smooth window
        self.WINDOWTYPE = 'TOPHAT'
        #options are 'TOPHAT', 'TOPHAT1D' and 'GAUSS' (for now). TOPHAT is calibrated for EPS, but GAUSS has less ringing

        # linear matter correlation functon smoothed over the same R0 (the LIM input) at z = 0 
        self.xi_linearmatter_smoothed_R0R0 = self.get_xi_R0_z0(Line_Parameters) 

# ---------------------
    # smoothed linear matter correlation function 
    def get_xi_R0_z0(self,  Line_Parameters):

        windowR = Window(self.WINDOWTYPE, self._klistCF, Line_Parameters._R) # only one value for the resolution but defined for array on the ks

        _Pksmooth = np.array(self._PklinCF) * windowR**2 

        self.rlist_CF, xi_R0R0_CF = self._xif(_Pksmooth, extrap = False) 

        return xi_R0R0_CF


class Power_Spectra_LIM:
    "Get LIM auto power spetrum from correlation functions and coefficients as function of z"

    def __init__(self, LIM_corr, LIM_coeff, Line_Parameters, Cosmo_Parameters, User_Parameters, RSD_MODE):

    ### STEP 0: Variable Setup
        self._r_CF = LIM_corr.rlist_CF # array of r for the correlation function

        self.klist_PS = LIM_corr._klistCF # array of k for the power spectrum 
        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        self.RSD_MODE = RSD_MODE # redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space 

        # linear growth factor
        self.lin_growth = cosmology.growth(Cosmo_Parameters, LIM_coeff.zintegral) 

    ### STEP 1: define the linear LIM power spectrum
        self.window_LIM = self.get_LIM_window(LIM_coeff, Line_Parameters)

        self._Pk_LIM_lin = (self.window_LIM.T * self.lin_growth)[:,np.newaxis]**2 * (Window(LIM_corr.WINDOWTYPE, LIM_corr._klistCF, Line_Parameters._R)**2 * LIM_corr._PklinCF)[np.newaxis,:]

        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2 

    ### STEP 2: define the cross LIM-delta linear power spectrum
        self._Pk_deltaLIM_lin = (self.window_LIM.T * self.lin_growth**2)[:,np.newaxis] * (Window(LIM_corr.WINDOWTYPE, LIM_corr._klistCF, Line_Parameters._R) *  LIM_corr._PklinCF)[np.newaxis,:] 
                                                                
        self.Deltasq_deltaLIM_lin = self._Pk_deltaLIM_lin * self._k3over2pi2 

    ### STEP 3: define the NON linear LIM power spectrum
        if Line_Parameters._R > User_Parameters.MAX_R_NONLINEAR:   
            self._Pk_LIM = self._Pk_LIM_lin
        else:
            # compute the correlations
            self.get_all_corrs_LIM(LIM_corr, LIM_coeff, Line_Parameters, Cosmo_Parameters, User_Parameters)

            self._Pk_LIM = self.get_list_PS(self._xiR0_LIM,  LIM_coeff.zintegral)
            self._Pk_LIM.T[:len(Cosmo_Parameters._Rtabsmoo)-Cosmo_Parameters.indexmaxNL] = self._Pk_LIM_lin.T[:len(Cosmo_Parameters._Rtabsmoo)-Cosmo_Parameters.indexmaxNL]            

        self.Deltasq_LIM = self._Pk_LIM * self._k3over2pi2 

    ### STEP 4: define the NON linear cross LIM-delta power spectrum assuming a lognormal for the delta
        if (User_Parameters.FLAG_DO_DENS_NL and Line_Parameters._R < User_Parameters.MAX_R_NONLINEAR):
            self._Pk_deltaLIM = self.get_list_PS(self._xiR0_deltaLIM, LIM_coeff.zintegral)

        else:
            self._Pk_deltaLIM = self._Pk_deltaLIM_lin
        
        if Line_Parameters.shot_noise:

            self.P_shot_noise = LIM_coeff.shot_noise[:,np.newaxis] * np.ones((len(LIM_coeff.zintegral),len(self.klist_PS)))

            self._Pk_LIM_tot = self._Pk_LIM + self.P_shot_noise

        else:
            self.P_shot_noise = 0.
            self._Pk_LIM_tot = self._Pk_LIM



    # --- #
    # define LIM window    
    def get_LIM_window(self, LIM_coeff, Line_Parameters):
        "Returns the LIM linear window function for all z in zintegral"

        gamma_R0 = LIM_coeff.gamma_LIM 

        # !!! move this one to UserParams
        if Line_Parameters.quadratic_lognormal:
            _win_LIM = LIM_coeff.Inu_bar * gamma_R0 / (1-2.*LIM_coeff.gamma2_LIM*LIM_coeff.sigmaofRtab_LIM**2)
        else:
            _win_LIM = LIM_coeff.Inu_bar * gamma_R0

        return _win_LIM
    
    # --- # 
    # get all the two point correlations
    def get_all_corrs_LIM(self, LIM_corr, LIM_coeff, Line_Parameters, Cosmo_Parameters, User_Parameters):
        "Returns the LIM components of the correlation functions of all observables at each z in zintegral"

        growthRmatrix = ((cosmology.growth(Cosmo_Parameters,LIM_coeff.zintegral))**2)[:,np.newaxis]
        
        gammaR0 = LIM_coeff.gamma_LIM[:,np.newaxis]
        sigmaR0 = LIM_coeff.sigmaofRtab_LIM[:,np.newaxis]
        g1 = (gammaR0 * sigmaR0)
        g2 = (gammaR0 * sigmaR0)

        xi_matter_R0_z0 = (LIM_corr.xi_linearmatter_smoothed_R0R0)[np.newaxis,:]

        xi_matter_R0_z = ne.evaluate('xi_matter_R0_z0 * growthRmatrix/ (sigmaR0 * sigmaR0)')
        xi_LIM_R0_z = ne.evaluate('g1 * g2 * xi_matter_R0_z')

        # !!! move this to User_Params
        if Line_Parameters.quadratic_lognormal:
            gammaR0_NL = LIM_coeff.gamma2_LIM[:,np.newaxis]
            g1NL = gammaR0_NL * sigmaR0**2
            g2NL = gammaR0_NL * sigmaR0**2

            numerator_NL = ne.evaluate('xi_LIM_R0_z + g1 * g1 * (0.5 - g2NL * (1 - xi_matter_R0_z * xi_matter_R0_z)) + g2 * g2 * (0.5 - g1NL * (1 - xi_matter_R0_z * xi_matter_R0_z))')
            
            denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - xi_matter_R0_z * xi_matter_R0_z)')
            
            norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)') 
            norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)') 
            
            log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')

            nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

        else:
            nonlinearcorrelation = ne.evaluate('exp(xi_LIM_R0_z)-1')

        self._xiR0_LIM = LIM_coeff.Inu_bar[:,np.newaxis]**2 * nonlinearcorrelation

        # --- #
        # if also matter treated as a smoothed lognormal
        if (User_Parameters.FLAG_DO_DENS_NL and Line_Parameters._R < User_Parameters.MAX_R_NONLINEAR):

            if Line_Parameters.quadratic_lognormal:

                g2 = sigmaR0
                g2NL = 0.

                numerator_NL = ne.evaluate('xi_LIM_R0_z+ g1 * g1 * (0.5 - g2NL * (1 - xi_matter_R0_z * xi_matter_R0_z)) + g2 * g2 * (0.5 - g1NL * (1 - xi_matter_R0_z * xi_matter_R0_z))')
                
                denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - xi_matter_R0_z * xi_matter_R0_z)')
                
                norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)') 
                norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)') 
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')

                nonlinear_deltaLIM = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

            else:
                xi_deltaLIM_R0_z = ne.evaluate('g1 * growthRmatrix * xi_matter_R0_z0')
                nonlinear_deltaLIM = ne.evaluate('exp(xi_deltaLIM_R0_z)-1')

            self._xiR0_deltaLIM = LIM_coeff.Inu_bar[:,np.newaxis] * nonlinear_deltaLIM

    # --- #
    # may be moved in the z21_utils
    def get_Pk_from_xi(self, rsinput, xiinput):
        "Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF"

        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf

    # may be moved in the z21_utils
    def get_list_PS(self, xi_list, zlisttoconvert):
        "Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input"

        _Pk_list = []
        for izp,zp in enumerate(zlisttoconvert):
            _kzp, _Pkzp = self.get_Pk_from_xi(self._r_CF,xi_list[izp])
            _Pk_list.append(_Pkzp)

        return np.array(_Pk_list)


# ----------------------------------------------------- #
# !!! to be moved in z21_utilities ? 
def _WinTH(k,R):
    x = k * R
    return 3.0/x**2 * (np.sin(x)/x - np.cos(x))

def _WinTH1D(k,R):
    x = k * R
    return  np.sin(x)/x

def _WinG(k,R):
    x = k * R * constants.RGauss_factor
    return np.exp(-x**2/2.0)

def Window(WINDOWTYPE, k, R):
    if WINDOWTYPE == 'TOPHAT':
        return _WinTH(k, R)
    elif WINDOWTYPE == 'GAUSS':
        return _WinG(k, R)
    elif WINDOWTYPE == 'TOPHAT1D':
        return _WinTH1D(k, R)
    else:
        print('ERROR in Window. Wrong type')