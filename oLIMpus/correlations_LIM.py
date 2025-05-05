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

    def __init__(self, Line_Parameters, Cosmo_Parameters, ClassCosmo,\
                    Line_Parameters_cross = None):

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

        # linear matter correlation functon smoothed over the same R1 for single line (the LIM input) at z = 0 , over two different lines for cross
        self.xi_linearmatter_smoothed_R1R2 = self.get_xi_R1R2_z0(Line_Parameters, Line_Parameters_cross) 

# ---------------------
    # smoothed linear matter correlation function 
    def get_xi_R1R2_z0(self,  Line_Parameters, Line_Parameters_cross):

        windowR = Window(self.WINDOWTYPE, self._klistCF, Line_Parameters._R) # only one value for the resolution but defined for array on the ks
        if Line_Parameters_cross is None:
            _Pksmooth = np.array(self._PklinCF) * windowR**2 
        else:
            windowR1 = Window(self.WINDOWTYPE, self._klistCF, Line_Parameters_cross._R)
            _Pksmooth = np.array(self._PklinCF) * windowR * windowR1 

        self.rlist_CF, xi_R1R2_CF = self._xif(_Pksmooth, extrap = False) 

        return xi_R1R2_CF



class Power_Spectra_LIM:
    "Get LIM auto power spetrum from correlation functions and coefficients as function of z"

    def __init__(self, LIM_corr, LIM_coeff, Line_Parameters, Cosmo_Parameters, User_Parameters, RSD_MODE,
                    LIM_coeff_cross = None, Line_Parameters_cross = None): # second line to cross correlate

    ### STEP 0: Variable Setup
        self._r_CF = LIM_corr.rlist_CF # array of r for the correlation function

        self.klist_PS = LIM_corr._klistCF # array of k for the power spectrum 
        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        self.RSD_MODE = RSD_MODE # redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space 

        # linear growth factor
        self.lin_growth = cosmology.growth(Cosmo_Parameters, LIM_coeff.zintegral) 

        ### STEP 1: define the linear LIM power spectrum
        self.window_LIM = self.get_LIM_window(LIM_coeff, Line_Parameters)
        if Line_Parameters_cross is None:
            self._Pk_LIM_lin = (self.window_LIM.T * self.lin_growth)[:,np.newaxis]**2 * (Window(LIM_corr.WINDOWTYPE, LIM_corr._klistCF, Line_Parameters._R)**2 * LIM_corr._PklinCF)[np.newaxis,:]

        ### STEP 2: define the cross LIM-delta linear power spectrum
            self._Pk_deltaLIM_lin = (self.window_LIM.T * self.lin_growth**2)[:,np.newaxis] * (Window(LIM_corr.WINDOWTYPE, LIM_corr._klistCF, Line_Parameters._R) *  LIM_corr._PklinCF)[np.newaxis,:] 

        else:
            self.window_LIM_cross = self.get_LIM_window(LIM_coeff_cross, Line_Parameters_cross)
    
        # in this case we have a cross power spectrum between lines and no cross with density
            self._Pk_LIM_lin = (self.window_LIM.T * self.window_LIM_cross.T * self.lin_growth**2)[:,np.newaxis] * (Window(LIM_corr.WINDOWTYPE, LIM_corr._klistCF, Line_Parameters._R)* (Window(LIM_corr.WINDOWTYPE, LIM_corr._klistCF, Line_Parameters_cross._R)) * LIM_corr._PklinCF)[np.newaxis,:]

            self._Pk_deltaLIM_lin = np.zeros_like(self._Pk_LIM_lin)

        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2                                                                 
        self.Deltasq_deltaLIM_lin = self._Pk_deltaLIM_lin * self._k3over2pi2 

    ### STEP 3: define the NON linear LIM power spectrum
        if Line_Parameters._R > User_Parameters.MAX_R_NONLINEAR:   
            self._Pk_LIM = self._Pk_LIM_lin
        else:
            # compute the correlations
            self.get_all_corrs_LIM(LIM_corr, LIM_coeff, Line_Parameters, Cosmo_Parameters, User_Parameters, \
                                   LIM_coeff_cross, Line_Parameters_cross)

            self._Pk_LIM = self.get_list_PS(self._xiR1R2_LIM,  LIM_coeff.zintegral)
            self._Pk_LIM.T[:len(Cosmo_Parameters._Rtabsmoo)-Cosmo_Parameters.indexmaxNL] = self._Pk_LIM_lin.T[:len(Cosmo_Parameters._Rtabsmoo)-Cosmo_Parameters.indexmaxNL]            

        self.Deltasq_LIM = self._Pk_LIM * self._k3over2pi2 

    ### STEP 4: define the NON linear cross LIM-delta power spectrum assuming a lognormal for the delta
        if (User_Parameters.FLAG_DO_DENS_NL and Line_Parameters._R < User_Parameters.MAX_R_NONLINEAR and Line_Parameters_cross is None):

            self._Pk_deltaLIM = self.get_list_PS(self._xiR1_deltaLIM, LIM_coeff.zintegral)
        else:
            self._Pk_deltaLIM = self._Pk_deltaLIM_lin
        
    ### STEP 5: sgot noise
        if Line_Parameters.shot_noise and Line_Parameters_cross is None:

            self.P_shot_noise = LIM_coeff.shot_noise[:,np.newaxis] * np.ones((len(LIM_coeff.zintegral),len(self.klist_PS)))

        else:
            self.P_shot_noise = 0.

        self._Pk_LIM_tot = self._Pk_LIM + self.P_shot_noise

    ### STEP 6: add RSD 
        if(self.RSD_MODE != 0): #with RSD (otherwise in real space)
            
            if(self.RSD_MODE==1): #spherically avg'd RSD
                mu2 = constants.MU_AVG**2 
            elif(self.RSD_MODE==2): #LoS RSD (mu=1)
                mu2 = constants.MU_LoS**2 
            else:
                print('Error, have to choose an RSD mode! RSD_MODE')

            dzlist = LIM_coeff.zintegral*0.001 
            # f(z) = dln D(d)/dln a = dln D(z) / dz * (dz/dln a)
            growth_rate = - (1.+LIM_coeff.zintegral) * (np.log(cosmology.growth(Cosmo_Parameters, LIM_coeff.zintegral+dzlist))-np.log(cosmology.growth(Cosmo_Parameters, LIM_coeff.zintegral-dzlist)))/(2.0*dzlist) 

            cross_RSD = 0. # !!! FIX 

            self._Pk_LIM_tot = self._Pk_LIM_tot + (growth_rate * mu2 * self.lin_growth[:,np.newaxis])**2 * LIM_corr._PklinCF + cross_RSD

    # --- #
    # define LIM window    
    def get_LIM_window(self, LIM_coeff, Line_Parameters):
        "Returns the LIM linear window function for all z in zintegral"

        gamma_R1 = LIM_coeff.gamma_LIM 

        # !!! move this one to UserParams
        if Line_Parameters.quadratic_lognormal:
            _win_LIM = LIM_coeff.Inu_bar * gamma_R1 / (1-2.*LIM_coeff.gamma2_LIM*LIM_coeff.sigmaofRtab_LIM**2)
        else:
            _win_LIM = LIM_coeff.Inu_bar * gamma_R1

        return _win_LIM
    
    # --- # 
    # get all the two point correlations
    def get_all_corrs_LIM(self, LIM_corr, LIM_coeff, Line_Parameters, Cosmo_Parameters, User_Parameters,\
                           LIM_coeff_cross, Line_Parameters_cross): # for line cross corr
        "Returns the LIM components of the correlation functions of all observables at each z in zintegral"

        growthRmatrix = ((cosmology.growth(Cosmo_Parameters,LIM_coeff.zintegral))**2)[:,np.newaxis]
        
        gammaR1 = LIM_coeff.gamma_LIM[:,np.newaxis]
        sigmaR1 = LIM_coeff.sigmaofRtab_LIM[:,np.newaxis]
        g1 = (gammaR1 * sigmaR1)
        if Line_Parameters_cross is None:
            g2 = (gammaR1 * sigmaR1)
            sigmaR2 = sigmaR1
        else:
            gammaR2 = LIM_coeff_cross.gamma_LIM[:,np.newaxis]
            sigmaR2 = LIM_coeff_cross.sigmaofRtab_LIM[:,np.newaxis]
            g2 = (gammaR2 * sigmaR2)

        xi_matter_R1R2_z0 = (LIM_corr.xi_linearmatter_smoothed_R1R2)[np.newaxis,:]

        xi_matter_R1R2_z = ne.evaluate('xi_matter_R1R2_z0 * growthRmatrix/ (sigmaR1 * sigmaR2)')
        xi_LIM_R1R2_z = ne.evaluate('g1 * g2 * xi_matter_R1R2_z')

        # !!! move this to User_Params
        if Line_Parameters.quadratic_lognormal:
            gammaR1_NL = LIM_coeff.gamma2_LIM[:,np.newaxis]
            g1NL = gammaR1_NL * sigmaR1**2
            if Line_Parameters_cross is None:
                g2NL = gammaR1_NL * sigmaR1**2
            else:
                gammaR2_NL = LIM_coeff_cross.gamma2_LIM[:,np.newaxis]
                g2NL = gammaR2_NL * sigmaR2**2

            numerator_NL = ne.evaluate('xi_LIM_R1R2_z + g1 * g1 * (0.5 - g2NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z)) + g2 * g2 * (0.5 - g1NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z))')
            
            denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z)')
            
            norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)') 
            norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)') 
            
            log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')

            nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

        else:
            nonlinearcorrelation = ne.evaluate('exp(xi_LIM_R1R2_z)-1')

        if Line_Parameters_cross is None:
            self._xiR1R2_LIM = LIM_coeff.Inu_bar[:,np.newaxis]**2 * nonlinearcorrelation
        else:
            self._xiR1R2_LIM = (LIM_coeff.Inu_bar * LIM_coeff_cross.Inu_bar)[:,np.newaxis] * nonlinearcorrelation

        # --- #
        # if also matter treated as a smoothed lognormal
        if (User_Parameters.FLAG_DO_DENS_NL and Line_Parameters._R < User_Parameters.MAX_R_NONLINEAR and Line_Parameters_cross is None):

            if Line_Parameters.quadratic_lognormal:

                g2 = sigmaR1
                g2NL = 0.

                numerator_NL = ne.evaluate('xi_LIM_R1R2_z+ g1 * g1 * (0.5 - g2NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z)) + g2 * g2 * (0.5 - g1NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z))')
                
                denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z)')
                
                norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)') 
                norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)') 
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')

                nonlinear_deltaLIM = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

            else:
                xi_deltaLIM_R1_z = ne.evaluate('g1 * growthRmatrix * xi_matter_R1R2_z')
                nonlinear_deltaLIM = ne.evaluate('exp(xi_deltaLIM_R1_z)-1')

            self._xiR1_deltaLIM = LIM_coeff.Inu_bar[:,np.newaxis] * nonlinear_deltaLIM
        else:
            self._xiR1_deltaLIM = 0.


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