'''
Model luminosity density and observed intensity for star forming lines.

Author: Sarah Libanore
BGU - April 2025
'''
from oLIMpus import constants
from oLIMpus import cosmology
from oLIMpus import sfrd, inputs_LIM
from oLIMpus import LIM_luminosities as L
import numpy as np 
import astropy.units as u 
import astropy.constants as cu 
from scipy.integrate import simpson

# Define the coefficients to be used in the LIM auto spectra computation; zmin is down to which the computation is performed
class get_LIM_coefficients:

    def __init__(self, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, User_Parameters, zmin): 

    ### STEP 0: Defining Constants and storage variables

        self.zmax_integral = constants.ZMAX_INTEGRAL
        self.zmin = zmin 

        self._dlogzint_target = 0.02/User_Parameters.precisionboost
        self.Nzintegral = np.ceil(1.0 + np.log(self.zmax_integral/self.zmin)/self._dlogzint_target).astype(int)
        self.dlogzint = np.log(self.zmax_integral/self.zmin)/(self.Nzintegral-1.0) #exact value rather than input target above

        self.zintegral = np.logspace(np.log10(self.zmin), np.log10(self.zmax_integral), self.Nzintegral)

        # compute sigmaR for the required resolution and redshift array
        self.sigmaofRtab_LIM = np.array([HMF_interpolator.sigmaR_int(Line_Parameters._R, zz) for zz in self.zintegral]).T[0]

        # EPS factors 
        Nsigmad = 1.0 #how many sigmas we explore
        Nds = 3 #how many deltas
        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)

    ### STEP 1: compute the average luminosity density in Lagrangian space

        zLIM_longer = np.geomspace(self.zmin, 50, 128) #extend to z = 50 for extrapolation purposes
        zLIM, mArray_LIM = np.meshgrid(zLIM_longer, HMF_interpolator.Mhtab, indexing = 'ij', sparse = True)

        # average luminosity density in Lagrangian space
        rhoL_avg_longer = np.trapz(rhoL_integrand(False, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zLIM), HMF_interpolator.logtabMh, axis = 1) 

        rhoL_interp = sfrd.interpolate.interp1d(zLIM_longer, rhoL_avg_longer, kind = 'cubic', bounds_error = False, fill_value = 0,) 

        # in the z array of interest
        self.rhoL_avg = rhoL_interp(self.zintegral) # Lagrangian ST
        self.rhoL_bar = rhoL_interp(self.zintegral) # this will be converted to EPS and to Eulerian

    ### STEP 2: Broadcasted Prescription to compute gammas

        # ---- #
        # resize arrays
        zArray_LIM, mArray_LIM, deltaNormArray_LIM = np.meshgrid(self.zintegral, HMF_interpolator.Mhtab, deltatab_norm, indexing = 'ij', sparse = True)

        sigmaR_LIM = self.sigmaofRtab_LIM[:,np.newaxis,np.newaxis]

        # ---- #
        # get sigma_M
        sigmaM_LIM = HMF_interpolator.sigmaintlog((np.log(mArray_LIM), zArray_LIM))

        # ---- #
        # compute the EPS correction
        modSigmaSq_LIM = sigmaM_LIM**2 - sigmaR_LIM**2
        indexTooBig = (modSigmaSq_LIM <= 0.0)
        modSigmaSq_LIM[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigmaSq_LIM = np.sqrt(modSigmaSq_LIM)

        nu0 = Cosmo_Parameters.delta_crit_ST / sigmaM_LIM # this is needed in the HMF 
        nu0[indexTooBig] = 1.0

        deltaArray_LIM = deltaNormArray_LIM * sigmaR_LIM

        modd_LIM = Cosmo_Parameters.delta_crit_ST - deltaArray_LIM
        nu = modd_LIM / modSigmaSq_LIM # used in the HMF

        EPS_HMF_corr_Lag = (nu/nu0) * (sigmaM_LIM/modSigmaSq_LIM)**2.0 * np.exp(-Cosmo_Parameters.a_corr_EPS * (nu**2-nu0**2)/2.0 ) 

        EPS_HMF_corr = (1.0 + deltaArray_LIM) * EPS_HMF_corr_Lag

        # ---- #
        # get the correct mean accounting for EPS 
        integrand_LIM_Lag = EPS_HMF_corr_Lag * rhoL_integrand(False, Line_Parameters,Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zArray_LIM)
        self.rhoL_dR_Lag = np.trapz(integrand_LIM_Lag, HMF_interpolator.logtabMh, axis = 1)

        # get the correct mean accounting for EPS and Eulerian
        integrand_LIM = EPS_HMF_corr * rhoL_integrand(False, Line_Parameters,Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zArray_LIM)
        self.rhoL_dR = np.trapz(integrand_LIM, HMF_interpolator.logtabMh, axis = 1)


        # ---- #
        # compute the gammas for the lognormal approximation as the derivatives of rhoL in Eulerian space
        midpoint = deltaArray_LIM.shape[-1]//2 

        self.gamma_LIM_Lag = np.log(self.rhoL_dR_Lag[:,midpoint+1]/self.rhoL_dR_Lag[:,midpoint-1]) / (deltaArray_LIM[:,0,midpoint+1] - deltaArray_LIM[:,0,midpoint-1])
        self.gamma_LIM_Lag[np.isnan(self.gamma_LIM_Lag)] = 0.0            

        self.gamma_LIM = np.log(self.rhoL_dR[:,midpoint+1]/self.rhoL_dR[:,midpoint-1]) / (deltaArray_LIM[:,0,midpoint+1] - deltaArray_LIM[:,0,midpoint-1])
        self.gamma_LIM[np.isnan(self.gamma_LIM)] = 0.0            

        x0 = deltaArray_LIM[:,0,midpoint-1]
        x1 = deltaArray_LIM[:,0,midpoint]
        x2 = deltaArray_LIM[:,0,midpoint+1]

        y0 = np.log(self.rhoL_dR[:,midpoint-1])
        y1 = np.log(self.rhoL_dR[:,midpoint])
        y2 = np.log(self.rhoL_dR[:,midpoint+1])
        self.gamma2_LIM = 2.*(y0/((x1-x0)*(x2-x0)) + y2/((x2-x1)*(x2-x0)) - y1/((x2-x1)*(x1-x0))) / 2.
        self.gamma2_LIM[np.isnan(self.gamma2_LIM)] = 0.0

        if Line_Parameters.quadratic_lognormal:
            self.norm_exp = np.exp((self.gamma_LIM * self.sigmaofRtab_LIM)**2/(2-4*self.gamma2_LIM*self.sigmaofRtab_LIM**2)) / np.sqrt(1-2*self.gamma2_LIM*self.sigmaofRtab_LIM**2)
        else:
            self.norm_exp = np.exp((self.gamma_LIM * self.sigmaofRtab_LIM)**2/2)

    ### STEP 3: Correct Eulerian-Lagrangian mean
        if(User_Parameters.C2_RENORMALIZATION_FLAG==True): # !!! move this flag to User_Params

            sigma = self.sigmaofRtab_LIM**2

            gamma_Lag = self.gamma_LIM_Lag
            #gamma_Lag = self.gamma_LIM -1

            #gamma_LIM_Lagrangian = self.gamma_LIM-1.0
            if Line_Parameters.quadratic_lognormal: # !!! move this flag to User_Params
                y0_Lag = np.log(self.rhoL_dR_Lag[:,midpoint-1])
                y1_Lag = np.log(self.rhoL_dR_Lag[:,midpoint])
                y2_Lag = np.log(self.rhoL_dR_Lag[:,midpoint+1])
                self.gamma2_LIM_Lag = 2.*(y0_Lag/((x1-x0)*(x2-x0)) + y2_Lag/((x2-x1)*(x2-x0)) - y1_Lag/((x2-x1)*(x1-x0))) / 2.
                self.gamma2_LIM_Lag[np.isnan(self.gamma2_LIM_Lag)] = 0.0

                gamma2_Lag = self.gamma2_LIM_Lag
                #gamma2_Lag = self.gamma2_LIM -1

                _corrfactorEulerian_LIM = (1+(gamma_Lag-2*gamma2_Lag)*sigma**2)/(1-2*gamma2_Lag*sigma**2) 
                #_corrfactorEulerian_LIM = self.norm_exp / (np.exp((self.gamma_LIM_Lag * self.sigmaofRtab_LIM)**2/(2-4*self.gamma2_LIM_Lag*self.sigmaofRtab_LIM**2)) / np.sqrt(1-2*self.gamma2_LIM_Lag*self.sigmaofRtab_LIM**2))
                

            else:
                _corrfactorEulerian_LIM =  1+ gamma_Lag * sigma**2
                #_corrfactorEulerian_LIM = self.norm_exp / np.exp((self.gamma_LIM_Lag * self.sigmaofRtab_LIM)**2/2)
                
            self.rhoL_bar *= _corrfactorEulerian_LIM

    ### STEP 4: Line Intensity Anisotropies
        if Line_Parameters.OBSERVABLE_LIM == 'Tnu':

            # c1 = uK / Lsun * Mpc^3
            self.coeff1_LIM = (((constants.c_kms * u.km/u.s)**3 * (1+self.zintegral)**2 / (8*np.pi * (cosmology.Hub(Cosmo_Parameters, self.zintegral) * u.km/u.s/u.Mpc) * (Line_Parameters.nu_rest)**3 * cu.k_B)).to(u.uK * u.Mpc**3 / u.Lsun )).value
            
        elif Line_Parameters.OBSERVABLE_LIM == 'Inu':

            # c1 = cm / sr / Hz so once is multiplied by rhoL gives Jy/sr
            if Line_Parameters.LINE_MODEL == 'SFRD':
                self.coeff1_LIM = np.ones(len(self.zintegral))
            else:
                self.coeff1_LIM = ((constants.c_kms * u.km/u.s / (4*np.pi *u.steradian) / (cosmology.Hub(Cosmo_Parameters, self.zintegral) * u.km/u.s/u.Mpc) / (Line_Parameters.nu_rest) * u.Lsun / u.Mpc**3).to(u.Jy/u.steradian)).value
        else:
            print('\nCHECK OBSERVABLE FOR LIM!')
            self.coeff1_LIM = -1

        # this is the observed intensity
        self.Inu_bar = self.coeff1_LIM * self.rhoL_bar

        if Line_Parameters.shot_noise:

            integrand_shot = P_shot_noise_integrand(False, Line_Parameters,Astro_Parameters, Cosmo_Parameters, HMF_interpolator, HMF_interpolator.Mhtab[np.newaxis,:], self.zintegral[:,np.newaxis])
            
            if Line_Parameters.OBSERVABLE_LIM == 'Tnu':

                scale_power_spectrum = ((self.coeff1_LIM * u.uK * u.Mpc**3 / u.Lsun)**2*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.uK**2)
            
            elif Line_Parameters.OBSERVABLE_LIM == 'Inu':

                scale_power_spectrum = (((self.coeff1_LIM*u.Jy/u.steradian/u.Lsun/u.Mpc**-3)**2)*u.Lsun**2*u.Mpc**-3).to(u.Mpc**3 * u.Jy**2/u.steradian**2)
            
            self.shot_noise = scale_power_spectrum.value * np.trapezoid(integrand_shot, HMF_interpolator.logtabMh, axis = 1) 

            self.shot_noise *= _corrfactorEulerian_LIM**2


### ---------------------- ###
# Integrand to compute the luminosity density
def rhoL_integrand(dotM, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z):
    "Integrand to compute the average line luminosity density"

    Mh = massVector # in Msun

    HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(Mh), z))) # in Mpc-3 Msun-1 

    Ltab_curr = LineLuminosity(dotM, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, z) 

    integrand_LIM = HMF_curr * Ltab_curr * Mh # in Lsun / Mpc3

    return integrand_LIM


def P_shot_noise_integrand(dotM, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z):
    "Integrand to compute the average line luminosity density"

    Mh = massVector # in Msun

    HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(Mh), z))) # in Mpc-3 

    dMdlogM = Mh
    dndlogM = HMF_curr * dMdlogM

    Ltab_curr = LineLuminosity(dotM, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, z) 

    integrand_P_shot_noise = dndlogM**-1 * (dndlogM * Ltab_curr)**2  # units Lsun2 Mpc-3 because of the delta Dirac ? 

    return integrand_P_shot_noise


### ---------------------- ###
# Compute the line luminosity from the SFR-Mh
def LineLuminosity(dotM, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z):
    "Luminosity-SFR-Mh relation for star forming lines. Units: Lsun"

    # check that all flags are compatible
    if Cosmo_Parameters.USE_RELATIVE_VELOCITIES or Cosmo_Parameters.Flag_emulate_21cmfast:
        print('\VCB OR EMULATE 21CMF IN COSMO PARAMS, NOT YET COMPATIBLE WITH OLIMPUS IMPLEMENTATION')
        return -1

    if Astro_Parameters.USE_POPIII or Astro_Parameters.USE_LW_FEEDBACK:
        print('\nPOPIII OR LW IN ASTRO PARAMS, NOT YET COMPATIBLE WITH OLIMPUS IMPLEMENTATION')
        return -1

    # --- #   
    # if not given as input, compute the SFR 
    if dotM is False:
        dotM = sfrd.SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z, z)    

    # --- #
    # line luminosity computation
    # 1) equal to the SFR
    if Line_Parameters.LINE_MODEL == 'SFRD':
        log10_L = np.log10(dotM)

    # 2) 
    else:
        if Line_Parameters.LINE_MODEL == 'Yang21':
            log10_L = getattr(L,Line_Parameters.LINE_MODEL)(Line_Parameters.LINE, massVector, z)
        elif Line_Parameters.LINE_MODEL == 'Lagache18':
            log10_L = getattr(L,Line_Parameters.LINE_MODEL)(Line_Parameters.LINE, dotM, z)
        else:
            try:
                log10_L = getattr(L,Line_Parameters.LINE_MODEL)(Line_Parameters.LINE, dotM)
            except:
                print('\nLINE MODEL NOT IMPLEMENTED')
                return -1

    # --- #
    # stochasticity computation
    if Line_Parameters.sigma_LMh == 0.:
        L_of_Mh = 10.**log10_L
    else:

        sigma_L = Line_Parameters.sigma_LMh
        if Line_Parameters.LINE_MODEL == 'Li16':
            sigma_L = (Line_Parameters.sigma_LMh**2 + (inputs_LIM.Li16_C021_params['sig_SFR'].value*np.log(10))**2/inputs_LIM.Li16_C021_params['alpha']**2)**0.5

        log_muL = np.log(10**log10_L) 
        log_muL[abs(log10_L) == np.inf] = 0.

        if len(log_muL.shape) == 2 or len(log_muL.shape) == 1:
            Lval = np.logspace(-50,20,503)[:,np.newaxis,np.newaxis]
        elif len(log_muL.shape) == 3:
            Lval = np.logspace(-50,20,503)[:,np.newaxis,np.newaxis,np.newaxis]

        coef = 1/(np.sqrt(2*np.pi)*sigma_L*Lval)

        # lognormal distribution

        p_logL =  coef * np.exp(- (np.log(Lval)-log_muL[np.newaxis,:])**2/(2*(sigma_L)**2))
        p_logL = np.where(np.isnan(p_logL), 0, p_logL)
        p_logL[p_logL < 1e-50] = 0.

        L_of_Mh = simpson(p_logL * Lval, Lval, axis=0)


    return L_of_Mh
