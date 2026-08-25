"""

Model luminosity density and observed intensity for star forming lines.

Author: Sarah Libanore
BGU - June 2026

"""

from oLIMpus.inputs_LIM import *
from oLIMpus import luminosities_LIM as L
from zeus21.sfrd import Z_init, SFRD_class
from scipy.integrate import quad_vec

class get_LIM_coefficients:
    """
    Define the coefficients to be used in the LIM auto- and cross- spectra computation.

    Parameters
    ----------
    UserParams : object
        User-defined parameters
    CosmoParams : object
        Cosmological parameters
    AstroParams : object
        Astrophysical parameters
    LineParams : object
        Line parameters
    HMFinterp : object
        Halo mass function interpolator
    z_Init : Z_init class, optional
        Initial redshift matrices for the calculation (see sfrd.py in zeus21 for details).
        Default is None.
    SFRD_Init : SFRD_class class, optional
        Initial star formation rate density for the calculation (see sfrd.py in zeus21 for details). 
        Default is None.

    Attributes
    ----------

    """

    def __init__(self, UserParams, CosmoParams, AstroParams, LineParams, HMFinterp, z_Init = None, SFRD_Init = None):

        # if z_Init and SFRD_Init are not provided, we initialize them here. This allows us to avoid redundant computations if they were already initialized in the parent class and passed as arguments.
        if z_Init is None:
            # to perform cross-correlation studies, the redshift array has to be the same as in zeus21
            self.z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams) 
        else:
            self.z_Init = z_Init

        if SFRD_Init is None:
            self.SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init) 
        else: 
            self.SFRD_Init = SFRD_Init
            
        self.USE_POPIII = AstroParams.USE_POPIII 

        # compute sigmaR for the required resolution and redshift array
        self.sigmaofRtab_LIM = np.array([HMFinterp.sigmaR_int(LineParams._R, zz) for zz in self.z_Init.zintegral]).T[0]

        # compute the average luminosity density in Lagrangian space
        zLIM_longer = np.geomspace(UserParams.zmin, constants.zmax_AstroBreak, 128) #e xtend to z = constants.zmax_AstroBreak for extrapolation purposes
        zLIM, mArray_LIM = np.meshgrid(zLIM_longer, HMFinterp.Mhtab, indexing = 'ij', sparse = True)

        # average luminosity density in Lagrangian space
        rhoL_avg_longer = np.trapezoid(self.rhoL_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zLIM), HMFinterp.logtabMh, axis = 1) 

        # correction to the average SFRD specific for Li16 model of CO 
        if LineParams.LINE_MODEL == 'Li16':

            if LineParams.line_dict is None:
                if LineParams.LINE == 'CO21':
                    line_dict = Li16_C021_params
                elif LineParams.LINE == 'CO10':
                    line_dict = Li16_C010_params
            else:
                line_dict = LineParams.line_dict

            rhoL_avg_longer *= np.exp((line_dict['alpha']**-2 - line_dict['alpha']**-1) * line_dict['sigma_SFR'].value**2 * np.log(10)**2 / 2.)

        rhoL_interp = sfrd.interpolate.interp1d(zLIM_longer, rhoL_avg_longer, kind = 'cubic', bounds_error = False, fill_value = 0,) 

        self.rhoL_avg = rhoL_interp(self.z_Init.zintegral) # Lagrangian
        self.rhoL_bar =  rhoL_interp(self.z_Init.zintegral) # this will be converted to EPS and to Eulerian

        self.compute_gamma_LIM(CosmoParams, AstroParams, LineParams, HMFinterp)

        # Correct Eulerian-Lagrangian mean
        if(UserParams.C2_RENORMALIZATION_FLAG==True): 

            sigma = self.sigmaofRtab_LIM**2

            gamma_Lag = self.gamma_LIM_Lag
            gamma2_Lag = self.gamma2_LIM_Lag

            if LineParams.quadratic_rhoL: 

                self._corrfactorEulerian_LIM = (1+(gamma_Lag-2*gamma2_Lag)*sigma**2)/(1-2*gamma2_Lag*sigma**2) 
                
            else:
                self._corrfactorEulerian_LIM =  1+ gamma_Lag * sigma**2
                
            self.rhoL_bar *= self._corrfactorEulerian_LIM

        # Line Intensity Anisotropies
        if LineParams.OBSERVABLE_LIM == 'Tnu':

            # c1 = uK / Lsun * Mpc^3
            self.coeff1_LIM = (((constants.c_kms * u.km/u.s)**3 * (1+self.z_Init.zintegral)**2 / (8*np.pi * (cosmology.Hub(CosmoParams, self.z_Init.zintegral) * u.km/u.s/u.Mpc) * (LineParams.nu_rest)**3 * cu.k_B)).to(u.uK * u.Mpc**3 / u.Lsun )).value
            
        elif LineParams.OBSERVABLE_LIM == 'Inu':

            # c1 = cm / sr / Hz so once is multiplied by rhoL gives Jy/sr
            self.coeff1_LIM = ((constants.c_kms * u.km/u.s / (4*np.pi *u.steradian) / (cosmology.Hub(CosmoParams, self.z_Init.zintegral) * u.km/u.s/u.Mpc) / (LineParams.nu_rest) * u.Lsun / u.Mpc**3).to(u.Jy/u.steradian)).value

            if LineParams.LINE == 'SFRD':
                self.coeff1_LIM = 1.
        else:
            print('\nCHECK OBSERVABLE FOR LIM!')
            self.coeff1_LIM = -1

        # this is the observed intensity
        self.Inu_bar = self.coeff1_LIM * self.rhoL_bar

        if LineParams.BURSTY_FLAG:
            self.sigPS_at_M(LineParams, HMFinterp.Mhtab[np.newaxis,:])
            self.VarL_bursty(LineParams)

        if LineParams.shot_noise:

            integrand_shot = self.P_shot_noise_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, HMFinterp.Mhtab[np.newaxis,:], self.z_Init.zintegral[:,np.newaxis])
            
            if LineParams.OBSERVABLE_LIM == 'Tnu':

                scale_power_spectrum = ((self.coeff1_LIM * u.uK * u.Mpc**3 / u.Lsun)**2*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.uK**2)
            
            elif LineParams.OBSERVABLE_LIM == 'Inu':

                scale_power_spectrum = (((self.coeff1_LIM*u.Jy/u.steradian/u.Lsun/u.Mpc**-3)**2)*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.Jy**2/u.steradian**2)
            
            self.shot_noise = scale_power_spectrum.value * np.trapezoid(integrand_shot, HMFinterp.logtabMh, axis = 1) 

            if(UserParams.C2_RENORMALIZATION_FLAG==True): 
                self.shot_noise *= self._corrfactorEulerian_LIM**2


    def compute_sigmaR_nu_LIM(self, CosmoParams, HMFinterp, z_array, Mh_array, d_array):

        zArray_LIM, mArray_LIM, deltaNormArray_LIM = np.meshgrid(z_array, Mh_array, d_array, indexing = 'ij', sparse = True)

        sigmaR_LIM = self.sigmaofRtab_LIM[:,np.newaxis,np.newaxis]

        # get sigma_M
        sigmaM_LIM = HMFinterp.sigmaintlog((np.log(mArray_LIM), zArray_LIM))

        # ---- #
        # compute the EPS correction
        modSigmaSq_LIM = sigmaM_LIM**2 - sigmaR_LIM**2
        indexTooBig = (modSigmaSq_LIM <= 0.0)
        modSigmaSq_LIM[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigmaSq_LIM = np.sqrt(modSigmaSq_LIM)

        nu0 = CosmoParams.delta_crit_ST / sigmaM_LIM # this is needed in the HMF 
        nu0[indexTooBig] = 1.0

        deltaArray_LIM = deltaNormArray_LIM * sigmaR_LIM

        modd_LIM = CosmoParams.delta_crit_ST - deltaArray_LIM
        nu = modd_LIM / modSigmaSq_LIM # used in the HMF

        EPS_HMF_corr_Lag = (nu/nu0) * (sigmaM_LIM/modSigmaSq_LIM)**2.0 * np.exp(-CosmoParams.a_corr_EPS * (nu**2-nu0**2)/2.0 ) 

        return EPS_HMF_corr_Lag, mArray_LIM, zArray_LIM, deltaArray_LIM
    

    def compute_gamma_LIM(self, CosmoParams, AstroParams, LineParams, HMFinterp):

        # EPS factors 
        Nsigmad = 1.0 # how many sigmas we explore
        Nds = 3 # how many deltas
        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)

        EPS_HMF_corr_Lag, mArray_LIM, zArray_LIM, deltaArray_LIM = self.compute_sigmaR_nu_LIM(CosmoParams, HMFinterp, self.z_Init.zintegral, HMFinterp.Mhtab, deltatab_norm)

        EPS_HMF_corr = (1.0 + deltaArray_LIM) * EPS_HMF_corr_Lag

        # get the correct mean accounting for EPS 
        integrand_LIM_Lag = EPS_HMF_corr_Lag * self.rhoL_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zArray_LIM)
        self.rhoL_dR_Lag = np.trapezoid(integrand_LIM_Lag, HMFinterp.logtabMh, axis = 1)

        # get the correct mean accounting for EPS and Eulerian
        integrand_LIM = EPS_HMF_corr * self.rhoL_integrand(False,CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zArray_LIM)
        self.rhoL_dR = np.trapezoid(integrand_LIM, HMFinterp.logtabMh, axis = 1)

        # compute the gammas for the lognormal approximation as the derivatives of rhoL in Eulerian space -- the function is defined in sfrd.py in zeus21
        self.gamma_LIM_Lag = SFRD_class.compute_numerical_der_gamma(SFRD_class, self.rhoL_dR_Lag[np.newaxis,:], deltaArray_LIM[np.newaxis,:], 1)[0]
        
        self.gamma_LIM = SFRD_class.compute_numerical_der_gamma(SFRD_class, self.rhoL_dR[np.newaxis,:], deltaArray_LIM[np.newaxis,:], 1)[0]

        self.gamma2_LIM_Lag = SFRD_class.compute_numerical_der_gamma(SFRD_class, self.rhoL_dR_Lag[np.newaxis,:], deltaArray_LIM[np.newaxis,:], 2)[0]
        
        self.gamma2_LIM = SFRD_class.compute_numerical_der_gamma(SFRD_class, self.rhoL_dR[np.newaxis,:], deltaArray_LIM[np.newaxis,:], 2)[0]
        
        if LineParams.quadratic_rhoL:
            self.norm_exp = np.exp((self.gamma_LIM * self.sigmaofRtab_LIM)**2/(2-4*self.gamma2_LIM*self.sigmaofRtab_LIM**2)) / np.sqrt(1-2*self.gamma2_LIM*self.sigmaofRtab_LIM**2)
        else:
            self.norm_exp = np.exp((self.gamma_LIM * self.sigmaofRtab_LIM)**2/2)

        return 1


    # Integrand to compute the luminosity density
    def rhoL_integrand(self, dotM, CosmoParams, AstroParams, LineParams, HMFinterp, massVector, z):
        "Integrand to compute the average line luminosity density"

        Mh = massVector # in Msun

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z))) # in Mpc-3 Msun-1 

        Ltab_curr = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams, HMFinterp, Mh, z) 

        integrand_LIM = HMF_curr * Ltab_curr * Mh # in Lsun / Mpc3

        return integrand_LIM


    def P_shot_noise_integrand(self, dotM, CosmoParams, AstroParams, LineParams, HMFinterp, massVector, z):
        "Integrand to compute the average line luminosity density"

        Mh = massVector # in Msun

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z))) # in Mpc-3 

        dMdlogM = Mh
        dndlogM = HMF_curr * dMdlogM

        Ltab_curr = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams, HMFinterp, Mh, z) 

        integrand_P_shot_noise = dndlogM * Ltab_curr**2  # units Lsun2 Mpc-3 because of the delta Dirac ? 

        if LineParams.BURSTY_FLAG:

            bursty_boost = 1. + self.V_lambda_burst
            integrand_P_shot_noise *= bursty_boost

        else:
                
            # sigma AT z: the scatter is not necessarily constant in redshift.
            integrand_P_shot_noise *= np.exp(LineParams.sigma_LMh_at(z)**2)

            if LineParams.LINE_MODEL == 'Li16':
                if LineParams.line_dict is None:
                    if LineParams.LINE == 'CO21':
                        line_dict = Li16_C021_params
                    elif LineParams.LINE == 'CO10':
                        line_dict = Li16_C010_params
                else:
                    line_dict = LineParams.line_dict

                integrand_P_shot_noise *= np.exp((2.*line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2)

        return integrand_P_shot_noise


    def P_shot_noise_cross_integrand(self, dotM, CosmoParams, AstroParams, LineParams, LineParams_cross, HMFinterp, massVector, z, cov_ln):
        "Integrand to compute the average line luminosity density"

        Mh = massVector # in Msun

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z))) # in Mpc-3 

        dMdlogM = Mh
        dndlogM = HMF_curr * dMdlogM

        Ltab_curr_1 = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams, HMFinterp, Mh, z) 
        Ltab_curr_2 = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams_cross, HMFinterp, Mh, z) 

        integrand_P_shot_noise = dndlogM * Ltab_curr_1 * Ltab_curr_2  # units Lsun2 Mpc-3 because of the delta Dirac ? 

        if LineParams.BURSTY_FLAG:

            bursty_boost = self.Cov_l1l2_burst
            integrand_P_shot_noise *= bursty_boost

        else:
                
            if cov_ln is None:
                cov_ln = (LineParams.sigma_LMh_at(z)
                          * LineParams_cross.sigma_LMh_at(z))  # rho = 1
            elif callable(cov_ln):
                # a redshift-dependent covariance, evaluated on the same grid
                cov_ln = cov_ln(z)

            integrand_P_shot_noise *= np.exp(cov_ln)

            if LineParams.LINE_MODEL == 'Li16':
                if LineParams.line_dict is None:
                    if LineParams.LINE == 'CO21':
                        line_dict = Li16_C021_params
                    elif LineParams.LINE == 'CO10':
                        line_dict = Li16_C010_params
                else:
                    line_dict = LineParams.line_dict

                integrand_P_shot_noise *= np.exp((line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2)

            if LineParams_cross.LINE_MODEL == 'Li16':
                if LineParams_cross.line_dict is None:
                    if LineParams_cross.LINE == 'CO21':
                        line_dict = Li16_C021_params
                    elif LineParams_cross.LINE == 'CO10':
                        line_dict = Li16_C010_params
                else:
                    line_dict = LineParams_cross.line_dict

                integrand_P_shot_noise *= np.exp((line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2)

        return integrand_P_shot_noise


    def LineLuminosity(self, dotM, CosmoParams, AstroParams, LineParams, HMFinterp, massVector, z):
        "Luminosity-SFR-Mh relation for star forming lines. Units: Lsun"

        # check that all flags are compatible
        if CosmoParams.USE_RELATIVE_VELOCITIES or CosmoParams.Flag_emulate_21cmfast:
            print('\VCB OR EMULATE 21CMF IN COSMO PARAMS, NOT YET COMPATIBLE WITH OLIMPUS IMPLEMENTATION')
            return -1

        if AstroParams.USE_POPIII :
            print('\nPOPIII OR LW IN ASTRO PARAMS, NOT YET COMPATIBLE WITH OLIMPUS IMPLEMENTATION')
            return -1

        # --- #   
        # if not given as input, compute the SFR 
        if dotM is False:
            if LineParams.LINE_MODEL == 'reproduce_Gong17':
                zval = np.array((1.,1.4,1.8,2.2,2.7,3.3,4.,4.8))
                aval = np.array((-7.90,-7.70,-7.5,-7.1,-6.78,-6.30,-6.15,-5.90))
                bval = np.array((  2.5, 2.49,2.49,2.42,2.36, 2.25, 2.25, 2.25))
                cval = np.array((-2.18,-2.18,-2.25,-2.10,-2.2,-2.2,-2.2,-2.2))

                a_z = interp1d(zval,aval,bounds_error=False,fill_value=(aval[0],aval[-1]))(z)
                b_z = interp1d(zval,bval,bounds_error=False,fill_value=(bval[0],bval[-1]))(z)
                c_z = interp1d(zval,cval,bounds_error=False,fill_value=(cval[0],cval[-1]))(z)

                dotM = 10**a_z * (massVector / 1e8)**b_z * (1. + massVector / 4e11)**c_z
            
            elif LineParams.LINE_MODEL == 'reproduce_Ambrose26':
                zfstar = np.array((5,6,7,8))
                fstarvals = np.array((0.021,0.024,0.031,0.061))
                fstar10 = interp1d(zfstar,fstarvals,bounds_error=False,fill_value=(fstarvals[0],fstarvals[-1]))(z)

                fstar = fstar10 * (massVector / 1e10)**AstroParams.alphastar

                Mstar = fstar * CosmoParams.OmegaB / CosmoParams.OmegaM * massVector

                tstar = 0.5
                dotM = Mstar * cosmology.Hubinvyr(CosmoParams,z) / tstar
            
            else:
                dotM = self.SFRD_Init.SFR(CosmoParams, AstroParams, HMFinterp, massVector, z, 2, False, False)    

        # --- #
        # line luminosity computation
        if LineParams.LINE_MODEL == 'Yang21':
            log10_L = getattr(L,LineParams.LINE_MODEL)(LineParams.LINE, massVector, z, LineParams.line_dict)        
        elif LineParams.LINE_MODEL == 'COMAP_fiducial':
            log10_L = getattr(L,LineParams.LINE_MODEL)(LineParams.LINE, massVector, LineParams.nu_rest, LineParams.line_dict)
        elif LineParams.LINE_MODEL == 'Lagache18':
            log10_L = getattr(L,LineParams.LINE_MODEL)(LineParams.LINE, dotM, z, LineParams.line_dict)
        elif LineParams.LINE_MODEL == 'reproduce_Ambrose26':
            log10_L = getattr(L,LineParams.LINE_MODEL)(z, CosmoParams, AstroParams, LineParams, massVector)
        elif LineParams.LINE_MODEL == 'reproduce_Gong17':
            log10_L = getattr(L,LineParams.LINE_MODEL)(z, LineParams, dotM)
        elif LineParams.LINE_MODEL == 'JWST_calibrated':
            log10_L = getattr(L, LineParams.LINE_MODEL)(LineParams.LINE, dotM, z, LineParams.line_dict)        
        else:
            try:
                log10_L = getattr(L, LineParams.LINE_MODEL)(LineParams.LINE, dotM, LineParams.line_dict)
            except:
                print('\nLINE MODEL NOT IMPLEMENTED')
                return -1

        # --- #
        # stochasticity computation
        if LineParams.LINE_MODEL == 'Li16':
            if LineParams.line_dict is None:
                line_dict = Li16_C021_params
            else:
                line_dict = LineParams.line_dict

            sigma_L = (LineParams.sigma_LMh**2 + (np.log(10) * line_dict['sigma_SFR'].value/line_dict['alpha'])**2)**0.5

        else:
            sigma_L = LineParams.sigma_LMh_at(z)

        # `sigma_L == 0.` was a scalar-only test and raised once sigma_L
        # became an array. Dropping it changes nothing: exp(0) = 1, so the
        # 'median' branch already reproduces the zero-scatter result.
        if LineParams.stoch_type == 'mean':
            L_of_Mh = 10.**log10_L
        
        else:        
            
            #  for a pure lognormal the numerical integration from the old version is unnecessary , the mean is analytic
            L_of_Mh = 10.**log10_L * np.exp(sigma_L**2 / 2.)

        L_of_Mh[dotM < 1e-20] = 0.

        return L_of_Mh


    # used if burstiness is on 
    def sigPS_at_M(self, LineParams, massVector):
        
        sigPS = LineParams.sigPS_piv_bursty + LineParams.dsigPS_dlog10M_bursty * (np.log10(massVector) - LineParams.log10M_piv_bursty)

        self.sPS = np.maximum(sigPS, LineParams.sigPS_cap_bursty)

        self.sx2 = 0.5 * self.sPS**2

        return 1


    def VarL_bursty(self, LineParams):
            
        integrand = lambda s: (LineParams.t_Myr_per_line - s) * (np.exp(self.sx2 * np.exp(-s / LineParams.tauPS_Myr_bursty)) - 1.0)

        V_top_hat, _ = quad_vec(integrand, 0.0, LineParams.t_Myr_per_line, limit=200, epsabs=1e-10, epsrel=1e-8)

        self.V_lambda_burst = V_top_hat * 2. / LineParams.t_Myr_per_line**2

        return 1


    def CovL_bursty(self, LineParams, LineParams_cross):

        def L(val):
            if val >= 0:
                return min(LineParams.t_Myr_per_line - val, LineParams_cross.t_Myr_per_line) if val < LineParams.t_Myr_per_line else 0.0
            else:
                return min(LineParams_cross.t_Myr_per_line + val, LineParams.t_Myr_per_line) if val > -LineParams_cross.t_Myr_per_line else 0.0
            
        integrand = lambda val: L(val) * (np.exp(self.sx2 * np.exp(-abs(val) / LineParams.tauPS_Myr_bursty)) - 1.0)

        val_neg, _ = quad_vec(integrand, -LineParams_cross.t_Myr_per_line, 0.0, limit=200, epsabs=1e-12, epsrel=1e-9)
        val_pos, _ = quad_vec(integrand,  0.0, LineParams.t_Myr_per_line, limit=200, epsabs=1e-12, epsrel=1e-9)

        self.Cov_l1l2_burst = (val_neg + val_pos) / (LineParams.t_Myr_per_line * LineParams_cross.t_Myr_per_line)

        return 1


    def __getattr__(self, name):
        """
        Access the attributes of the classes that we initialized directly from the get_T21_coefficients class, without having to specify which class they come from

        Parameters
        ----
        name: str
            Name of the attribute to get
        
        Returns
        -------
        attribute            
        The attribute with the given name, if it exists in any of the classes that we initialized. If it does not exist in any of them, raises an AttributeError.
        """

        list_of_cls = [self.z_Init, self.SFRD_Init]

        if self.USE_POPIII:
            list_of_cls += [self.relvel]
        for cls in list_of_cls:
            try:
                return getattr(cls, name)
            except AttributeError:
                pass

        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
