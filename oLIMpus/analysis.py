
from oLIMpus import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from zeus21.inputs import User_Parameters, Cosmo_Parameters, Astro_Parameters
from zeus21 import T21coefficients, correlations, Z_init, SFRD_class
from oLIMpus.inputs_LIM import Line_Parameters
from copy import copy

plt.rcParams.update({"text.usetex": True, "font.family": "Times new roman"}) # Use latex fonts
colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) # Set the color palette as default
plt.rcParams['legend.frameon'] = False

plt.rcParams["figure.figsize"] = (5.7, 4.5)
plt.rcParams['lines.linewidth'] = 2
fontsize = 15
plt.rcParams.update({
    "font.size": fontsize,              # general font size
    "axes.titlesize": fontsize,          # title font size
    "axes.labelsize": fontsize,          # x and y label size
    "xtick.labelsize": fontsize-3,          # x tick label size
    "ytick.labelsize": fontsize-3,          # y tick label size
    "legend.fontsize": fontsize,          # legend font size
    "figure.titlesize": fontsize         # figure title font size
})


CosmoParams_inputs = dict(
        omegab= 0.0223828,
        omegac = 0.1201075,
        h_fid = 0.67810,
        As = 2.100549e-09,
        ns = 0.9660499,
        tau_fid = 0.05430842,
        HMF_CHOICE= "ST",
        Flag_emulate_21cmfast = False,
        )

AstroParams_inputs = dict(
        # values from 2306.09403
        # alphastar = 0.61,
        # betastar = -1.91,
        # epsstar = 0.1, # pivot at z = 8
        # Mc = 10**12.03,
        # dlog10epsstardz = -0.0467,

        # # we fix these values to get fduty == 1
        # Mturn_fixed = 1e-10,
        accretion_model = 'exp',
        alphastar = 0.5,
        betastar = -0.5,
        epsstar = 0.1,
        Mc = 3e11,
        Mturn_fixed = None,
        dlog10epsstardz = 0.0,
        quadratic_SFRD_lognormal = True,
        USE_LW_FEEDBACK = False,

        fesc10 = 0.1,
        alphaesc = 0.,
        L40_xray = 3.0,
        E0_xray = 500.,
        alpha_xray = -1.0,
        Emax_xray_norm=2000,

        )



"Class to store the quantities needed in the LIM computation and analysis, define in the input list the ones that you want to vary while the others are fiducial"
class run_oLIMpus:

    def __init__(self, LINE, LINE_MODEL = 'Yang24', _R = 1., shot_noise= False, quadratic_lognormal=True, sigma_LMh = 0., RSD_MODE = 0, SIGMA_FOG=0.,\
        alphastar = 0.5,
        betastar = -0.5,
        epsstar = 0.1,
        Mc = 3e11,
        Mturn_fixed = None,
        dlog10epsstardz = 0.0,
        fesc=0.1,
        LIM_observable = 'Inu',
        line_dict = None):

        self.UP = User_Parameters(
            precisionboost= 1.0,
            FLAG_FORCE_LINEAR_CF=  False,
            MIN_R_NONLINEAR= 0.5,
            MAX_R_NONLINEAR= 200.0,
            FLAG_DO_DENS_NL= False,
            FLAG_WF_ITERATIVE= True,
            )

        print('Setting Cosmology...')
        self.CP = Cosmo_Parameters(UserParams=self.UP, **CosmoParams_inputs)

        print('...Defining HMF...')
        self.HMFcl = cosmology.HMF_interpolator(UserParams=self.UP,CosmoParams=self.CP)

        print('...Setting Astrophysics')
        AstroParams_input = copy(AstroParams_inputs)
        AstroParams_input['quadratic_SFRD_lognormal'] = quadratic_lognormal
        AstroParams_input['alphastar'] = alphastar
        AstroParams_input['betastar'] = betastar
        AstroParams_input['epsstar'] = epsstar
        AstroParams_input['Mturn_fixed'] = Mturn_fixed
        AstroParams_input['Mc'] = Mc
        AstroParams_input['dlog10epsstardz'] = dlog10epsstardz
        AstroParams_input['fesc10'] = fesc

        self.AP = Astro_Parameters(CosmoParams= self.CP, **AstroParams_input)

        print('...Setting Line properties...')
        self.LP = Line_Parameters(
            LINE = LINE, # which line
            LINE_MODEL = LINE_MODEL, # model of the line luminosity
            OBSERVABLE_LIM = LIM_observable, # observe intensity in Jy/sr or mK
            _R = _R, # resolution for smoothing
            sigma_LMh_dex = sigma_LMh, # FIX 10: sigma_LMh is now an init=False field
            shot_noise = shot_noise, # add shot noise to the power spectrum
            quadratic_rhoL = quadratic_lognormal, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
            line_dict= line_dict # parameters that enter the L-SFR or L-Mh relation
        )

        print('...Initiating SFRD class...')
        self.z_Init = Z_init(UserParams=self.UP, CosmoParams=self.CP)
        self.SFRD_Init = SFRD_class(UserParams=self.UP, CosmoParams=self.CP, AstroParams=self.AP, HMFinterp=self.HMFcl, z_Init=self.z_Init)

        print('...Running Line...')
        self.LIMcoeff = coefficients_LIM.get_LIM_coefficients(self.UP,self.CP,self.AP,self.LP,self.HMFcl,z_Init=self.z_Init,SFRD_Init=self.SFRD_Init)

        self.LIMpk = correlations_LIM.Power_Spectra_LIM(self.UP,self.CP,self.LP,self.LIMcoeff,RSD_MODE=RSD_MODE,SIGMA_FOG=SIGMA_FOG)

        print('...Running T21...')
        self.T21coeff = T21coefficients.get_T21_coefficients(self.UP,self.CP,self.AP,self.HMFcl, z_Init=self.z_Init,SFRD_Init=self.SFRD_Init)

        self.T21pk = correlations.Power_Spectra(self.UP,self.CP,self.AP,self.T21coeff, RSD_MODE=RSD_MODE)

        print('...Done!')
