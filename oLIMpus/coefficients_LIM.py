"""

Model luminosity density and observed intensity for star forming lines.

Author: Sarah Libanore
BGU - June 2026

"""

from oLIMpus.inputs_LIM import *
from oLIMpus import luminosities_LIM as L
from oLIMpus import burstiness_LIM
from zeus21.sfrd import Z_init, SFRD_class

# delta/sigma_R grid used only to normalise C_EPS. +-6 sigma is well past where the
# Gaussian weight matters; 161 points converges the normalisation to better than 1e-6.
_DELTA_NORM_GRID = np.linspace(-6., 6., 161)


def EPS_HMF_ratio(sigmaM, sigmaR, delta, delta_crit, a_corr):
    """Eq. 6 of arXiv:2507.15922: (dn_EPS/dMh)(delta_R) / <dn_EPS/dMh>.

    Identically zero wherever sigma_M < sigma_R: a halo whose Lagrangian radius exceeds
    R does not fit inside the smoothing sphere and is cut from the sum.
    """
    modSigmaSq = sigmaM**2 - sigmaR**2
    tooBig = modSigmaSq <= 0.0
    modSigma = np.sqrt(np.where(tooBig, np.inf, modSigmaSq))
    nu0 = np.where(tooBig, 1.0, delta_crit / sigmaM)
    nu = (delta_crit - delta) / modSigma

    return (nu/nu0) * (sigmaM/modSigma)**2.0 * np.exp(-a_corr * (nu**2 - nu0**2)/2.0)


def EPS_HMF_norm(sigmaM, sigmaR, delta_crit, a_corr):
    """<C_EPS>_delta_R, the factor that makes Eq. 5 do what its text says it does.

    The paragraph under Eq. 5 of arXiv:2507.15922 states that rescaling the EPS ratio by
    the Sheth-Tormen HMF recovers the correct average when integrating over Mh, i.e.
    <C_EPS>_delta = dn/dMh. That identity is exact for a_corr = 1 (pure Press-Schechter)
    and is broken by a_ST = 0.707: measured at z = 6, R0 = 1 Mpc it is 1.05 at 1e8 Msun
    but 0.53 at 1e11, which propagates straight into rho_L_bar through Eq. 8. Dividing
    the ratio by this factor enforces the identity.

    Returns 1 where the ratio vanishes, so halos that do not fit in R stay excluded
    rather than being divided by zero. The residual left there is not a normalisation:
    those halos are outside the reach of the conditional HMF at that R.
    """
    d = _DELTA_NORM_GRID * sigmaR
    ratio = EPS_HMF_ratio(sigmaM, sigmaR, d, delta_crit, a_corr)
    w = np.exp(-_DELTA_NORM_GRID**2/2.) / np.sqrt(2*np.pi)
    n = np.trapezoid(ratio * w, _DELTA_NORM_GRID, axis=-1)[..., np.newaxis]

    return np.where(n > 0., n, 1.)


def _resolve_Li16_dict(LineParams):
    """The Li16 parameter dictionary for this line.

    Was duplicated in four places (the mean-SFRD correction and the three shot-noise
    branches); one helper keeps them from drifting apart.
    """
    if LineParams.line_dict is not None:
        return LineParams.line_dict
    if LineParams.LINE == 'CO21':
        return Li16_C021_params
    if LineParams.LINE == 'CO10':
        return Li16_C010_params
    raise ValueError("LINE_MODEL='Li16' has no default parameters for LINE=%r; "
                     "pass line_dict explicitly." % LineParams.LINE)


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
        # HMFinterp.sigmaR_int builds a (1,2) array per z; sigmaRintlog is the
        # underlying RegularGridInterpolator and takes the whole z column.
        _zz = self.z_Init.zintegral
        self.sigmaofRtab_LIM = HMFinterp.sigmaRintlog(
            (np.full_like(_zz, np.log(LineParams._R)), _zz))

        # compute the average luminosity density in Lagrangian space
        zLIM_longer = np.geomspace(UserParams.zmin, constants.zmax_AstroBreak, 128) #e xtend to z = constants.zmax_AstroBreak for extrapolation purposes
        zLIM, mArray_LIM = np.meshgrid(zLIM_longer, HMFinterp.Mhtab, indexing = 'ij', sparse = True)

        # average luminosity density in Lagrangian space
        rhoL_avg_longer = np.trapezoid(self.rhoL_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zLIM), HMFinterp.logtabMh, axis = 1) 

        # correction to the average SFRD specific for Li16 model of CO 
        if LineParams.LINE_MODEL == 'Li16':

            line_dict = _resolve_Li16_dict(LineParams)

            rhoL_avg_longer *= np.exp((line_dict['alpha']**-2 - line_dict['alpha']**-1) * line_dict['sigma_SFR'].value**2 * np.log(10)**2 / 2.)

        # bounds_error=True on purpose. With fill_value=0 a mismatch between
        # UserParams.zmin and the z_Init grid (e.g. reusing a zmin=5 UserParams with a
        # zmin=2.5 Z_init) silently returned rho_L = 0, and therefore Inu_bar = 0, for
        # every redshift below UserParams.zmin. Fail loudly instead.
        rhoL_interp = sfrd.interpolate.interp1d(zLIM_longer, rhoL_avg_longer, kind = 'cubic', bounds_error = True)

        self.rhoL_avg = rhoL_interp(self.z_Init.zintegral) # Lagrangian
        self.rhoL_bar =  rhoL_interp(self.z_Init.zintegral) # this will be converted to EPS and to Eulerian

        self.compute_gamma_LIM(CosmoParams, AstroParams, LineParams, HMFinterp)

        # Correct Eulerian-Lagrangian mean, Eqs. 13-14 of arXiv:2507.15922.
        if(UserParams.C2_RENORMALIZATION_FLAG==True):

            sigma2 = self.sigmaofRtab_LIM**2

            gamma_Lag = self.gamma_LIM_Lag
            gamma2_Lag = self.gamma2_LIM_Lag

            if LineParams.quadratic_rhoL:

                self._corrfactorEulerian_LIM = (1+(gamma_Lag-2*gamma2_Lag)*sigma2)/(1-2*gamma2_Lag*sigma2)

            else:
                self._corrfactorEulerian_LIM =  1+ gamma_Lag * sigma2

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
            # Burstiness IS the luminosity scatter in the mean-anchored convention of
            # arXiv:2605.13967. Stacking the phenomenological sigma_LMh on top of it, or
            # using stoch_type='median', would double count the second moment. Extra
            # NON-burstiness scatter goes through sigma_extra_dex instead.
            if LineParams.stoch_type != 'mean' or LineParams.sigma_LMh_dex != 0.:
                raise ValueError("BURSTY_FLAG=True requires stoch_type='mean' and "
                                 "sigma_LMh_dex=0: the OU burstiness already supplies "
                                 "the luminosity scatter (arXiv:2605.13967, Eq. 6). "
                                 "Use sigma_extra_dex for non-burstiness scatter.")
            if LineParams.LINE_MODEL == 'Li16':
                raise ValueError("BURSTY_FLAG=True is incompatible with LINE_MODEL='Li16', "
                                 "whose line_dict['sigma_SFR'] is itself a lognormal SFR "
                                 "scatter and would be double counted.")

            _Mh = HMFinterp.Mhtab[np.newaxis, :]
            self.sPS = burstiness_LIM.sigma_PS_of_M(_Mh, LineParams)
            self.V_lambda_burst = burstiness_LIM.V_lambda(_Mh, LineParams)
            self.boost_per_halo = burstiness_LIM.boost_per_halo(_Mh, LineParams)

        if LineParams.shot_noise:

            _Mh_col = HMFinterp.Mhtab[np.newaxis,:]
            _z_col = self.z_Init.zintegral[:,np.newaxis]

            # The deterministic (halo-discreteness) integrand, Eq. 3 of 2605.13967, and the
            # one that carries the scatter, Eq. 2. Keeping both makes the mass-integrated
            # boost B_lambda(z) a ratio the user can read off directly.
            integrand_shot_det = self.P_shot_noise_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, _Mh_col, _z_col, deterministic=True)
            integrand_shot = self.P_shot_noise_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, _Mh_col, _z_col)
            
            if LineParams.OBSERVABLE_LIM == 'Tnu':

                scale_power_spectrum = ((self.coeff1_LIM * u.uK * u.Mpc**3 / u.Lsun)**2*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.uK**2)
            
            elif LineParams.OBSERVABLE_LIM == 'Inu':

                scale_power_spectrum = (((self.coeff1_LIM*u.Jy/u.steradian/u.Lsun/u.Mpc**-3)**2)*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.Jy**2/u.steradian**2)
            
            self.shot_noise = scale_power_spectrum.value * np.trapezoid(integrand_shot, HMFinterp.logtabMh, axis = 1)
            self.shot_noise_det = scale_power_spectrum.value * np.trapezoid(integrand_shot_det, HMFinterp.logtabMh, axis = 1)

            if(UserParams.C2_RENORMALIZATION_FLAG==True): 
                self.shot_noise *= self._corrfactorEulerian_LIM**2
                self.shot_noise_det *= self._corrfactorEulerian_LIM**2


    def compute_sigmaR_nu_LIM(self, CosmoParams, HMFinterp, z_array, Mh_array, d_array,
                              normalize_CEPS=True):

        zArray_LIM, mArray_LIM, deltaNormArray_LIM = np.meshgrid(z_array, Mh_array, d_array, indexing = 'ij', sparse = True)

        sigmaR_LIM = self.sigmaofRtab_LIM[:,np.newaxis,np.newaxis]

        # get sigma_M
        sigmaM_LIM = HMFinterp.sigmaintlog((np.log(mArray_LIM), zArray_LIM))

        deltaArray_LIM = deltaNormArray_LIM * sigmaR_LIM

        EPS_HMF_corr_Lag = EPS_HMF_ratio(sigmaM_LIM, sigmaR_LIM, deltaArray_LIM,
                                         CosmoParams.delta_crit_ST, CosmoParams.a_corr_EPS)

        if normalize_CEPS:
            EPS_HMF_corr_Lag = EPS_HMF_corr_Lag / EPS_HMF_norm(
                sigmaM_LIM, sigmaR_LIM, CosmoParams.delta_crit_ST, CosmoParams.a_corr_EPS)

        return EPS_HMF_corr_Lag, mArray_LIM, zArray_LIM, deltaArray_LIM
    

    def compute_gamma_LIM(self, CosmoParams, AstroParams, LineParams, HMFinterp):

        # EPS factors 
        Nsigmad = 1.0 # how many sigmas we explore
        Nds = 3 # how many deltas
        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)

        EPS_HMF_corr_Lag, mArray_LIM, zArray_LIM, deltaArray_LIM = self.compute_sigmaR_nu_LIM(CosmoParams, HMFinterp, self.z_Init.zintegral, HMFinterp.Mhtab, deltatab_norm, LineParams.normalize_CEPS)

        # get the correct mean accounting for EPS 
        integrand_LIM_Lag = EPS_HMF_corr_Lag * self.rhoL_integrand(False, CosmoParams, AstroParams, LineParams, HMFinterp, mArray_LIM, zArray_LIM)
        self.rhoL_dR_Lag = np.trapezoid(integrand_LIM_Lag, HMFinterp.logtabMh, axis = 1)

        # EPS_HMF_corr = (1+delta) * EPS_HMF_corr_Lag and (1+delta) does not
        # depend on Mh, so the Eulerian integral is exactly (1+delta) times the
        # Lagrangian one. 
        self.rhoL_dR = (1.0 + deltaArray_LIM[:, 0, :]) * self.rhoL_dR_Lag

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


    def P_shot_noise_integrand(self, dotM, CosmoParams, AstroParams, LineParams, HMFinterp, massVector, z, deterministic=False):
        """Integrand of the shot-noise power spectrum, Eq. 35 of arXiv:2507.15922.

        dn/dlogMh * <L^2>(Mh, z). With `deterministic=True` the second moment is
        replaced by Lbar^2, i.e. the halo-discreteness term alone (Eq. 3 of
        arXiv:2605.13967); the ratio of the two integrals is the mass-integrated
        boost B_lambda.
        """

        Mh = massVector # in Msun

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z))) # in Mpc-3

        dMdlogM = Mh
        dndlogM = HMF_curr * dMdlogM

        Ltab_curr = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams, HMFinterp, Mh, z)

        integrand_P_shot_noise = dndlogM * Ltab_curr**2  # units Lsun2 Mpc-3 because of the delta Dirac ?

        if deterministic:
            return integrand_P_shot_noise

        if LineParams.BURSTY_FLAG:

            # <L^2>/Lbar^2 = 1 + V_lambda(Mh), Eq. 6 of arXiv:2605.13967
            integrand_P_shot_noise = integrand_P_shot_noise * burstiness_LIM.boost_per_halo(Mh, LineParams)

        else:

            # sigma AT z: the scatter is not necessarily constant in redshift.
            integrand_P_shot_noise = integrand_P_shot_noise * np.exp(LineParams.sigma_LMh_at(z)**2)

            if LineParams.LINE_MODEL == 'Li16':
                line_dict = _resolve_Li16_dict(LineParams)

                integrand_P_shot_noise = integrand_P_shot_noise * np.exp((2.*line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2)

        return integrand_P_shot_noise


    def P_shot_noise_cross_integrand(self, dotM, CosmoParams, AstroParams, LineParams, LineParams_cross, HMFinterp, massVector, z, cov_ln, deterministic=False):
        """Integrand of the cross shot-noise power spectrum between two lines.

        dn/dlogMh * <L1 L2>(Mh, z), Eq. 8 of arXiv:2605.13967. With
        `deterministic=True` the same-halo cross moment is replaced by Lbar1*Lbar2.
        """

        Mh = massVector # in Msun

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z))) # in Mpc-3

        dMdlogM = Mh
        dndlogM = HMF_curr * dMdlogM

        Ltab_curr_1 = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams, HMFinterp, Mh, z)
        Ltab_curr_2 = self.LineLuminosity(dotM, CosmoParams, AstroParams, LineParams_cross, HMFinterp, Mh, z)

        integrand_P_shot_noise = dndlogM * Ltab_curr_1 * Ltab_curr_2

        if deterministic:
            return integrand_P_shot_noise

        if LineParams.BURSTY_FLAG:

            # <L1 L2>/(Lbar1 Lbar2) = 1 + V_12(Mh), Eqs. 9-10 of arXiv:2605.13967.
            # The '1 +' is the deterministic halo-discreteness term and must not be dropped.
            integrand_P_shot_noise = integrand_P_shot_noise * burstiness_LIM.boost_per_halo_cross(Mh, LineParams, LineParams_cross)

        else:

            if cov_ln is None:
                cov_ln = (LineParams.sigma_LMh_at(z)
                          * LineParams_cross.sigma_LMh_at(z))  # rho = 1
            elif callable(cov_ln):
                # a redshift-dependent covariance, evaluated on the same grid
                cov_ln = cov_ln(z)

            integrand_P_shot_noise = integrand_P_shot_noise * np.exp(cov_ln)

            if LineParams.LINE_MODEL == 'Li16':
                line_dict = _resolve_Li16_dict(LineParams)
                integrand_P_shot_noise = integrand_P_shot_noise * np.exp((line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2)

            if LineParams_cross.LINE_MODEL == 'Li16':
                line_dict = _resolve_Li16_dict(LineParams_cross)
                integrand_P_shot_noise = integrand_P_shot_noise * np.exp((line_dict['alpha']**-2-line_dict['alpha']**-1)*line_dict['sigma_SFR'].value**2*np.log(10)**2)

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
            line_dict = _resolve_Li16_dict(LineParams)

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

        # L_of_Mh and dotM do not always have the same shape (e.g. models
        # that depend on Mh only, such as Yang21/COMAP_fiducial, return a mass-only
        # array while dotM is (nz, nM)). Broadcast before masking.
        L_of_Mh, dotM_b = np.broadcast_arrays(np.asarray(L_of_Mh, dtype=float), dotM)
        L_of_Mh = np.where(dotM_b < 1e-20, 0., L_of_Mh)

        return L_of_Mh


    # ----------------------------------------------------------------- #
    # Burstiness diagnostics (arXiv:2605.13967). The machinery itself lives in
    # burstiness_LIM.py; these are the observables the paper defines.
    # ----------------------------------------------------------------- #

    def B_lambda(self):
        """Mass-integrated shot-noise boost B_lambda(z), Eq. 7 of arXiv:2605.13967.

        The Lbar^2-weighted average of the per-halo boost over the halo mass function,
        i.e. simply P_shot / P_shot|deterministic on this code's own HMF and L(Mh).
        Array over zintegral. Equals 1 identically when BURSTY_FLAG is False and there
        is no lognormal scatter.
        """
        return self.shot_noise / self.shot_noise_det


    def sigma_L_equivalent_dex(self):
        """The mass-independent lognormal scatter that reproduces B_lambda(z), in dex.

        Standard LIM forecasts adopt sigma_L ~ 0.3-0.5 dex; this puts the burstiness
        prediction on the same axis. Array over zintegral.
        """
        return burstiness_LIM.sigma_L_equivalent_dex(self.B_lambda())


    def shot_noise_weight(self, CosmoParams, AstroParams, LineParams, HMFinterp, z,
                          LineParams_cross=None):
        """dn/dlogMh * Lbar1(Mh,z) * Lbar2(Mh,z): the weight of the shot-noise integral.

        This is the kernel that decides WHICH halos the boost is averaged over, so it
        is what you plot to see where B_lambda comes from, and what R_cross needs.
        Returns an array on HMFinterp.Mhtab.
        """
        Mh = HMFinterp.Mhtab[np.newaxis, :]
        zz = np.atleast_2d(np.asarray(z, dtype=float)).reshape(-1, 1)

        if LineParams_cross is None:
            w = self.P_shot_noise_integrand(False, CosmoParams, AstroParams, LineParams,
                                            HMFinterp, Mh, zz, deterministic=True)
        else:
            w = self.P_shot_noise_cross_integrand(False, CosmoParams, AstroParams, LineParams,
                                                  LineParams_cross, HMFinterp, Mh, zz,
                                                  cov_ln=None, deterministic=True)
        return np.squeeze(w)


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
