"""
Code to compute LIM correlation functions and power spectra. Based on correlations.py 

Author: Sarah Libanore
BGU - April 2025

"""

from oLIMpus.inputs_LIM import * 


class Power_Spectra_LIM:

    """
    Get the LIM power spectrum and its components from correlation functions and coefficients
    
    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    LineParams : LineParams class
    LIMcoeffs : LIMcoeffs class (see coefficients_LIM.py)
    LineParams_cross : second LineParams class to perform cross-correlation (default None)
    LIMcoeffs_cross : second LIMcoeffs class to perform cross-correlation (default None)
    RSD_MODE : int
        Choice of redshift-space distortion mode.
        0 = None (mu=0), just for comparison with real-space
        1 = Spherical avg (like 21-cmFAST), standard assumption in sims
        2 = LoS only (mu=1), more observationally relevant
        Default is 1
    SIGMA_FOG : float
        Fingers-of-God parameter (default = 0.)
    
    Attributes
    ----------
    Basic Setup Attributes

    """

    def __init__(self,  UserParams, CosmoParams, LineParams, LIMcoeffs, HMFinterp = None, AstroParams=None, LineParams_cross = None, LIMcoeffs_cross = None, cov_ln_cross=None, RSD_MODE = 1, SIGMA_FOG = 0.):
                 
        self._xi2P = None
        self.get_xi_R0(CosmoParams, LineParams, LineParams_cross)
        
        self.klist_PS = CosmoParams._klistCF # array of k for the power spectrum 
        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        self.RSD_MODE = RSD_MODE 

        # linear growth factor
        self.lin_growth = cosmology.growth(CosmoParams, LIMcoeffs.zintegral) 

        # define the linear LIM power spectrum
        self.window_LIM = self.get_LIM_window(LineParams, LIMcoeffs)
        if LineParams_cross is None:
            self._Pk_LIM_lin = (self.window_LIM.T * self.lin_growth)[:,np.newaxis]**2 * (z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE)**2 * CosmoParams._PklinCF)[np.newaxis,:]

            # define the cross LIM-delta linear power spectrum
            self._Pk_deltaLIM_lin = (self.window_LIM.T * self.lin_growth**2)[:,np.newaxis] * (z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE) *  CosmoParams._PklinCF)[np.newaxis,:] 

        else:
            self.window_LIM_cross = self.get_LIM_window(LineParams_cross, LIMcoeffs_cross)
    
            # in this case we have a cross power spectrum between lines and no cross with density
            self._Pk_LIM_lin = (self.window_LIM.T * self.window_LIM_cross.T * self.lin_growth**2)[:,np.newaxis] * (z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE)* (z21_utilities.Window(CosmoParams._klistCF, LineParams_cross._R, self.WINDOWTYPE)) * CosmoParams._PklinCF)[np.newaxis,:]

            self._Pk_deltaLIM_lin = (self.window_LIM.T * self.lin_growth**2)[:,np.newaxis] * (z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE) *  CosmoParams._PklinCF)[np.newaxis,:] 
            
            self._Pk_deltaLIM_lin_cross = (self.window_LIM_cross.T * self.lin_growth**2)[:,np.newaxis] * (z21_utilities.Window(CosmoParams._klistCF, LineParams_cross._R, self.WINDOWTYPE) *  CosmoParams._PklinCF)[np.newaxis,:] 

        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2                                                                 
        self.Deltasq_deltaLIM_lin = self._Pk_deltaLIM_lin * self._k3over2pi2 

        # define the NON linear LIM power spectrum
        if LineParams._R > UserParams.MAX_R_NONLINEAR:   
            self._Pk_LIM = self._Pk_LIM_lin
        else:
            # compute the correlations
            self.get_all_corrs_LIM(UserParams, CosmoParams, LineParams, LIMcoeffs, LineParams_cross, LIMcoeffs_cross)

            self._Pk_LIM = self.get_list_PS_fast(CosmoParams, self._xiR1R2_LIM)
            self._Pk_LIM.T[:len(CosmoParams._Rtabsmoo)-CosmoParams.indexmaxNL] = self._Pk_LIM_lin.T[:len(CosmoParams._Rtabsmoo)-CosmoParams.indexmaxNL]            


        self._Pk_LIM[self._Pk_LIM < 0.] = 0. # this is to avoid when using large smoothing that drops below 0
        self.Deltasq_LIM = self._Pk_LIM * self._k3over2pi2 

    # define the NON linear cross LIM-delta power spectrum assuming a lognormal for the delta
        # for _R >= MAX_R_NONLINEAR get_all_corrs_LIM is never called, so
        # _xiR1_deltaLIM does not exist and _Pk_deltaLIM was left undefined -> the
        # clipping line below raised AttributeError. Fall back to the linear result.
        if (LineParams._R < UserParams.MAX_R_NONLINEAR):
            self._Pk_deltaLIM = self.get_list_PS_fast(CosmoParams, self._xiR1_deltaLIM)
        else:
            self._Pk_deltaLIM = np.array(self._Pk_deltaLIM_lin)
        if LineParams_cross is not None:
            if (LineParams_cross._R < UserParams.MAX_R_NONLINEAR):
                self._Pk_deltaLIM_cross = self.get_list_PS_fast(CosmoParams, self._xiR1_deltaLIM_cross)
            else:
                self._Pk_deltaLIM_cross = np.array(self._Pk_deltaLIM_lin_cross)

        self._Pk_deltaLIM[self._Pk_deltaLIM < 0.] = 0. # this is to avoid when using large smoothing that drops below 0

    # add RSD             
        if(self.RSD_MODE==0): #spherically avg'd RSD
            mu2 = 0.
        elif(self.RSD_MODE==1): #spherically avg'd RSD
            mu2 = constants.MU_AVG**2 
        elif(self.RSD_MODE==2): #LoS RSD (mu=1)
            mu2 = constants.MU_LoS**2 
        else:
            print('Error, have to choose an RSD mode! RSD_MODE')

        dzlist = LIMcoeffs.zintegral*0.001 
        # f(z) = dln D(d)/dln a = dln D(z) / dz * (dz/dln a)
        growth_rate = - (1.+LIMcoeffs.zintegral) * (np.log(cosmology.growth(CosmoParams, LIMcoeffs.zintegral+dzlist))-np.log(cosmology.growth(CosmoParams, LIMcoeffs.zintegral-dzlist)))/(2.0*dzlist) 

        if LineParams_cross is None:
            self._Pk_LIM_RSD = self._Pk_LIM + LIMcoeffs.Inu_bar[:,np.newaxis]**2 * (growth_rate[:,np.newaxis] * mu2 * self.lin_growth[:,np.newaxis])**2 * CosmoParams._PklinCF[np.newaxis,:] + 2 * LIMcoeffs.Inu_bar[:,np.newaxis] * growth_rate[:,np.newaxis] * mu2 * self._Pk_deltaLIM
        else:
            self._Pk_LIM_RSD = self._Pk_LIM + LIMcoeffs.Inu_bar[:,np.newaxis] * LIMcoeffs_cross.Inu_bar[:,np.newaxis] * (growth_rate[:,np.newaxis] * mu2 * self.lin_growth[:,np.newaxis])**2 \
              * CosmoParams._PklinCF[np.newaxis,:] + LIMcoeffs_cross.Inu_bar[:,np.newaxis] * growth_rate[:,np.newaxis] * mu2 * self._Pk_deltaLIM \
              + LIMcoeffs.Inu_bar[:,np.newaxis] * growth_rate[:,np.newaxis] * mu2 * self._Pk_deltaLIM_cross
        if SIGMA_FOG != 0.:
            self._Pk_LIM_RSD /= (1.+ mu2*(self.klist_PS*SIGMA_FOG)**2/2.)**2

        self._Pk_LIM_RSD[self._Pk_LIM_RSD < 0.] = 0. # this is to avoid when using large smoothing that drops below 0

    ### STEP 6: shot noise
        if LineParams.shot_noise:

            if LineParams_cross is None:

                self.P_shot_noise = LIMcoeffs.shot_noise[:,np.newaxis] * np.ones((len(LIMcoeffs.zintegral),len(self.klist_PS)))
                
                self.P_shot_noise *= z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE)**2

            else:

                if LineParams.BURSTY_FLAG != LineParams_cross.BURSTY_FLAG:
                    raise ValueError('BURSTY_FLAG must match between the two lines.')

                # deterministic (halo-discreteness) and full same-halo cross moments
                integrand_shot_noise_cross_det = LIMcoeffs.P_shot_noise_cross_integrand(False,CosmoParams,AstroParams,LineParams,LineParams_cross,HMFinterp,HMFinterp.Mhtab[np.newaxis,:], LIMcoeffs.zintegral[:,np.newaxis], cov_ln=cov_ln_cross, deterministic=True)
                integrand_shot_noise_cross = LIMcoeffs.P_shot_noise_cross_integrand(False,CosmoParams,AstroParams,LineParams,LineParams_cross,HMFinterp,HMFinterp.Mhtab[np.newaxis,:], LIMcoeffs.zintegral[:,np.newaxis], cov_ln=cov_ln_cross)

                if LineParams.OBSERVABLE_LIM == 'Tnu':

                    scale_power_spectrum = np.sqrt(((LIMcoeffs.coeff1_LIM * u.uK * u.Mpc**3 / u.Lsun)**2*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.uK**2))
                
                elif LineParams.OBSERVABLE_LIM == 'Inu':

                    scale_power_spectrum = np.sqrt((((LIMcoeffs.coeff1_LIM*u.Jy/u.steradian/u.Lsun/u.Mpc**-3)**2)*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.Jy**2/u.steradian**2))

                if LineParams_cross.OBSERVABLE_LIM == 'Tnu':

                    scale_power_spectrum_cross = np.sqrt(((LIMcoeffs_cross.coeff1_LIM * u.uK * u.Mpc**3 / u.Lsun)**2*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.uK**2))
                
                elif LineParams_cross.OBSERVABLE_LIM == 'Inu':

                    scale_power_spectrum_cross = np.sqrt((((LIMcoeffs_cross.coeff1_LIM*u.Jy/u.steradian/u.Lsun/u.Mpc**-3)**2)*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.Jy**2/u.steradian**2))

                _scale = scale_power_spectrum.value * scale_power_spectrum_cross.value
                shot_noise_cross = _scale * np.trapezoid(integrand_shot_noise_cross, HMFinterp.logtabMh, axis = 1)
                shot_noise_cross_det = _scale * np.trapezoid(integrand_shot_noise_cross_det, HMFinterp.logtabMh, axis = 1)

                if (UserParams.C2_RENORMALIZATION_FLAG==True):
                    _phi = (LIMcoeffs._corrfactorEulerian_LIM * LIMcoeffs_cross._corrfactorEulerian_LIM)
                    shot_noise_cross *= _phi
                    shot_noise_cross_det *= _phi

                # kept as attributes so the cross boost, and hence the R diagnostic of
                # arXiv:2605.13967 Eq. 11, can be read straight off the released code
                self.shot_noise_cross = shot_noise_cross
                self.shot_noise_cross_det = shot_noise_cross_det

                self.P_shot_noise = shot_noise_cross[:,np.newaxis] * np.ones((len(LIMcoeffs.zintegral),len(self.klist_PS)))
                
                self.P_shot_noise *= (z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE)*z21_utilities.Window(CosmoParams._klistCF, LineParams_cross._R, self.WINDOWTYPE))

        else:

            self.P_shot_noise = 0.

        self._Pk_LIM_tot = self._Pk_LIM_RSD + self.P_shot_noise


    def R_cross(self, LIMcoeffs, LIMcoeffs_cross):
        """Cross-line correlation coefficient of the bursty shot-noise excess.

        Eq. 11 of arXiv:2605.13967:  R = Delta_12 / sqrt(Delta_1 Delta_2), with
        Delta_ij = P_shot^ij / P_shot^ij|det - 1. The deterministic term and all
        luminosity normalisations cancel, leaving a quantity that depends mainly on
        tau_PS. Requires a cross run (LineParams_cross given) with shot noise on.
        Array over zintegral.
        """
        d12 = self.shot_noise_cross / self.shot_noise_cross_det - 1.
        d1 = LIMcoeffs.shot_noise / LIMcoeffs.shot_noise_det - 1.
        d2 = LIMcoeffs_cross.shot_noise / LIMcoeffs_cross.shot_noise_det - 1.

        return d12 / np.sqrt(d1 * d2)

    
    # z21_utilities.get_list_PS rebuilds the whole mcfit transform inside a
    # python loop over redshift, even though rlist_CF is z-independent. Building it
    # once and letting mcfit act on the last axis of the (nz, nr) array is ~150x
    # faster and bit-for-bit identical.
    def get_list_PS_fast(self, CosmoParams, xi_list):
        if getattr(self, '_xi2P', None) is None:
            self._xi2P = mcfit.xi2P(CosmoParams.rlist_CF, l=0, lowring=True)
        _k, _Pk = self._xi2P(np.asarray(xi_list), extrap=False)
        return _Pk

    # define LIM window
    def get_LIM_window(self, LineParams, LIMcoeffs):
        "Returns the LIM linear window function for all z in zintegral"

        gamma_R1 = LIMcoeffs.gamma_LIM 

        # !!! move this one to UserParams
        if LineParams.quadratic_rhoL:
            _win_LIM = LIMcoeffs.Inu_bar * gamma_R1 / (1-2.*LIMcoeffs.gamma2_LIM*LIMcoeffs.sigmaofRtab_LIM**2)
        else:
            _win_LIM = LIMcoeffs.Inu_bar * gamma_R1

        return _win_LIM
    
    # --- # 
    # get all the two point correlations
    def get_all_corrs_LIM(self, UserParams, CosmoParams, LineParams, LIMcoeffs, LineParams_cross, LIMcoeffs_cross): # for line cross corr
        "Returns the LIM components of the correlation functions of all observables at each z in zintegral"

        growthRmatrix = ((cosmology.growth(CosmoParams,LIMcoeffs.zintegral))**2)[:,np.newaxis]
        
        gammaR1 = LIMcoeffs.gamma_LIM[:,np.newaxis]
        sigmaR1 = LIMcoeffs.sigmaofRtab_LIM[:,np.newaxis]
        g1 = (gammaR1 * sigmaR1)
        if LineParams_cross is None:
            g2 = (gammaR1 * sigmaR1)
            sigmaR2 = sigmaR1
        else:
            gammaR2 = LIMcoeffs_cross.gamma_LIM[:,np.newaxis]
            sigmaR2 = LIMcoeffs_cross.sigmaofRtab_LIM[:,np.newaxis]
            g2 = (gammaR2 * sigmaR2)

        xi_matter_R1R2_z0 = (self.xi_linearmatter_smoothed_R0)[np.newaxis,:]

        xi_matter_R1R2_z = ne.evaluate('xi_matter_R1R2_z0 * growthRmatrix/ (sigmaR1 * sigmaR2)')
        xi_LIM_R1R2_z = ne.evaluate('g1 * g2 * xi_matter_R1R2_z')

        # !!! move this to User_Params
        if LineParams.quadratic_rhoL:
            gammaR1_NL = LIMcoeffs.gamma2_LIM[:,np.newaxis]
            g1NL = gammaR1_NL * sigmaR1**2
            if LineParams_cross is None:
                g2NL = gammaR1_NL * sigmaR1**2
            else:
                gammaR2_NL = LIMcoeffs_cross.gamma2_LIM[:,np.newaxis]
                g2NL = gammaR2_NL * sigmaR2**2

            norm1 = LIMcoeffs.norm_exp[:,np.newaxis] 
            if LineParams_cross is None:
                norm2 = LIMcoeffs.norm_exp[:,np.newaxis]  
            else:
                norm2 = LIMcoeffs_cross.norm_exp[:,np.newaxis]

            numerator_NL = ne.evaluate('xi_LIM_R1R2_z + g1 * g1 * (0.5 - g2NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z)) + g2 * g2 * (0.5 - g1NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z))')
            
            denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - xi_matter_R1R2_z * xi_matter_R1R2_z)')
            
            log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')

            nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

        else:
            nonlinearcorrelation = ne.evaluate('(exp(xi_LIM_R1R2_z)-1)')

        if LineParams_cross is None:
            self._xiR1R2_LIM = LIMcoeffs.Inu_bar[:,np.newaxis]**2 * nonlinearcorrelation 

        else:
            self._xiR1R2_LIM = (LIMcoeffs.Inu_bar * LIMcoeffs_cross.Inu_bar)[:,np.newaxis] * nonlinearcorrelation

        # --- #
        # if also matter treated as a smoothed lognormal
        if (LineParams._R < UserParams.MAX_R_NONLINEAR ): 

            windowR1 = z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE) # only one value for the resolution but defined for array on the ks
            _Pksmooth = np.array(CosmoParams._PklinCF) * windowR1

            self.rlist_CF, xi_R10_CF = CosmoParams._xif(_Pksmooth, extrap = False) 
            xi_matter_R10_z0 = (xi_R10_CF)[np.newaxis,:]
            xi_matter_R10_z = ne.evaluate('xi_matter_R10_z0 * growthRmatrix')

            if LineParams.quadratic_rhoL:

                numerator_NL = ne.evaluate('gammaR1 * xi_matter_R10_z + gammaR1_NL * xi_matter_R10_z*xi_matter_R10_z + g1*g1/2')
                    
                denominator_NL = ne.evaluate('1. - 2 * g1NL')
                                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1)')
                
                nonlinear_deltaLIM = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

            else:
                nonlinear_deltaLIM = ne.evaluate('(exp(gammaR1*xi_matter_R10_z) -1)')

            self._xiR1_deltaLIM = LIMcoeffs.Inu_bar[:,np.newaxis] * nonlinear_deltaLIM

            if LineParams_cross is not None:

                windowR2_cross = z21_utilities.Window(CosmoParams._klistCF, LineParams_cross._R,self.WINDOWTYPE, ) # only one value for the resolution but defined for array on the ks
                _Pksmooth_cross = np.array(CosmoParams._PklinCF) * windowR2_cross

                self.rlist_CF, xi_R20_CF_cross = CosmoParams._xif(_Pksmooth_cross, extrap = False)
                xi_matter_R20_z0_cross = (xi_R20_CF_cross)[np.newaxis,:]
                xi_matter_R20_z_cross = ne.evaluate('xi_matter_R20_z0_cross * growthRmatrix')
         
                if LineParams.quadratic_rhoL:

                    numerator_NL = ne.evaluate('gammaR2 * xi_matter_R20_z_cross + gammaR2_NL * xi_matter_R20_z_cross*xi_matter_R20_z_cross + g2*g2/2')
                        
                    denominator_NL = ne.evaluate('1. - 2 * g2NL')
                                        
                    log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm2)')
                    
                    nonlinear_deltaLIM = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

                else:
                    nonlinear_deltaLIM = ne.evaluate('(exp(gammaR2*xi_matter_R20_z_cross)-1)')

                self._xiR1_deltaLIM_cross = LIMcoeffs_cross.Inu_bar[:,np.newaxis] * nonlinear_deltaLIM

        else:
            self._xiR1_deltaLIM = 0.
            if LineParams_cross is not None:
                self._xiR1_deltaLIM_cross = 0.


    def get_xi_R0(self, CosmoParams, LineParams, LineParams_cross):

        self.WINDOWTYPE = "TOPHAT"
        windowR = z21_utilities.Window(CosmoParams._klistCF, LineParams._R, self.WINDOWTYPE) # only one value for the resolution but defined for array on the ks
        if LineParams_cross is None:
            _Pksmooth = np.array(CosmoParams._PklinCF) * windowR**2 
        else:
            windowR1 = z21_utilities.Window(CosmoParams._klistCF, LineParams_cross._R, self.WINDOWTYPE)
            _Pksmooth = np.array(CosmoParams._PklinCF) * windowR * windowR1 

        self.rlist_CF, self.xi_linearmatter_smoothed_R0 = CosmoParams._xif(_Pksmooth, extrap = False) 

