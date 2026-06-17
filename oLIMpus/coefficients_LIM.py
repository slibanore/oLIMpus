"""

Model luminosity density and observed intensity for star forming lines.

Author: Sarah Libanore
BGU - June 2026

"""

from oLIMpus.inputs_LIM import *
from oLIMpus import luminosities_LIM as L
from zeus21.sfrd import Z_init, SFRD_class

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
        rhoL_avg_longer = np.trapz(self.rhoL_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zLIM), HMFinterp.logtabMh, axis = 1) 

        # correction to the average SFRD specific for Li16 model of CO 
        if LineParams.stoch_type == 'mean' and LineParams.LINE_MODEL == 'Li16':
            if LineParams.line_dict is None:
                if LineParams.LINE == 'CO21':
                    line_dict = Li16_C021_params
                elif LineParams.LINE == 'CO10':
                    line_dict = Li16_C010_params
            else:
                line_dict = LineParams.line_dict

            rhoL_avg_longer *= np.exp((line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2/2.)

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

                _corrfactorEulerian_LIM = (1+(gamma_Lag-2*gamma2_Lag)*sigma**2)/(1-2*gamma2_Lag*sigma**2) 
                
            else:
                _corrfactorEulerian_LIM =  1+ gamma_Lag * sigma**2
                
            self.rhoL_bar *= _corrfactorEulerian_LIM

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

        if LineParams.shot_noise:

            integrand_shot = self.P_shot_noise_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, HMFinterp.Mhtab[np.newaxis,:], self.z_Init.zintegral[:,np.newaxis])
            
            if LineParams.OBSERVABLE_LIM == 'Tnu':

                scale_power_spectrum = ((self.coeff1_LIM * u.uK * u.Mpc**3 / u.Lsun)**2*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.uK**2)
            
            elif LineParams.OBSERVABLE_LIM == 'Inu':

                scale_power_spectrum = (((self.coeff1_LIM*u.Jy/u.steradian/u.Lsun/u.Mpc**-3)**2)*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.Jy**2/u.steradian**2)
            
            self.shot_noise = scale_power_spectrum.value * np.trapezoid(integrand_shot, HMFinterp.logtabMh, axis = 1) 

            self.shot_noise *= _corrfactorEulerian_LIM**2


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
        self.rhoL_dR_Lag = np.trapz(integrand_LIM_Lag, HMFinterp.logtabMh, axis = 1)

        # get the correct mean accounting for EPS and Eulerian
        integrand_LIM = EPS_HMF_corr * self.rhoL_integrand(False,CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zArray_LIM)
        self.rhoL_dR = np.trapz(integrand_LIM, HMFinterp.logtabMh, axis = 1)

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

        integrand_P_shot_noise = dndlogM**-1 * (dndlogM * Ltab_curr)**2  # units Lsun2 Mpc-3 because of the delta Dirac ? 

        if LineParams.stoch_type == 'mean':
            
            integrand_P_shot_noise *= np.exp(LineParams.sigma_LMh.value**2*np.log(10)**2)

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
            dotM = self.SFRD_Init.SFR(CosmoParams, AstroParams, HMFinterp, massVector, z, 2, False, False)    

        # --- #
        # line luminosity computation
        if LineParams.LINE_MODEL == 'Yang21':
            log10_L = getattr(L,LineParams.LINE_MODEL)(LineParams.LINE, massVector, z, LineParams.line_dict)
        elif LineParams.LINE_MODEL == 'Lagache18':
            log10_L = getattr(L,LineParams.LINE_MODEL)(LineParams.LINE, dotM, z, LineParams.line_dict)
        else:
            try:
                log10_L = getattr(L,LineParams.LINE_MODEL)(LineParams.LINE, dotM, LineParams.line_dict)
            except:
                print('\nLINE MODEL NOT IMPLEMENTED')
                return -1

        # --- #
        # stochasticity computation
        if LineParams.sigma_LMh == 0. or LineParams.stoch_type == 'mean':
            L_of_Mh = 10.**log10_L
        else:
            if LineParams.LINE_MODEL == 'Li16':
                if LineParams.line_dict is None:
                    line_dict = Li16_C021_params
                else:
                    line_dict = LineParams.line_dict

                sigma_L = (LineParams.sigma_LMh.value**2 + (line_dict['sigma_SFR'].value/line_dict['alpha'])**2)**0.5

            else:
                sigma_L = LineParams.sigma_LMh.value
        
            sigma_L = sigma_L * np.log(10)

            log_muL = np.log(10**log10_L) 
            log_muL[abs(log10_L) == np.inf] = 0.

            if len(log_muL.shape) == 2 or len(log_muL.shape) == 1:
                Lval = np.logspace(-50,20,503)[:,np.newaxis,np.newaxis]
            elif len(log_muL.shape) == 3:
                Lval = np.logspace(-50,20,503)[:,np.newaxis,np.newaxis,np.newaxis]

            coef = 1./(np.sqrt(2*np.pi)*sigma_L*Lval)

            # lognormal distribution

            p_logL =  coef * np.exp(- (np.log(Lval)-log_muL[np.newaxis,:])**2/(2*(sigma_L)**2))
            p_logL = np.where(np.isnan(p_logL), 0, p_logL)
            p_logL[p_logL < 1e-50] = 0.

            L_of_Mh = simpson(p_logL * Lval, Lval, axis=0)

        L_of_Mh[dotM < 1e-20] = 0.

        return L_of_Mh


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
