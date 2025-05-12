from oLIMpus import * 
import matplotlib.pyplot as plt
import matplotlib as mpl
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


CosmoParams_input_fid = dict(
        omegab= 0.0223828, 
        omegac = 0.1201075, 
        h_fid = 0.67810, 
        As = 2.100549e-09, 
        ns = 0.9660499, 
        tau_fid = 0.05430842, 
        HMF_CHOICE= "ST",
        Flag_emulate_21cmfast = False,
        )

AstroParams_input_fid = dict(
        # values from 2306.09403
        # alphastar = 0.61,
        # betastar = -1.91,
        # epsstar = 0.1, # pivot at z = 8
        # Mc = 10**12.03,
        # dlog10epsstardz = -0.0467,

        # # we fix these values to get fduty == 1
        # Mturn_fixed = 1e-10,

        astromodel = 0,
        accretion_model = 0,
        alphastar = 0.5,
        betastar = -0.5,
        epsstar = 0.1,
        Mc = 3e11,
        Mturn_fixed = None,
        dlog10epsstardz = 0.0,
        quadratic_SFRD_lognormal = False, # Sarah Libanore

#############################
        fesc10 = 0.1,
        alphaesc = 0.,
        L40_xray = 3.0,
        E0_xray = 500.,
        alpha_xray = -1.0,
        Emax_xray_norm=2000,

        Nalpha_lyA_II = 9690,
        Nalpha_lyA_III = 17900,

        FLAG_MTURN_SHARP= False,

        C0dust = 4.43,
        C1dust = 1.99,
        sigmaUV=0.5,

        USE_POPIII = False,
        alphastar_III = 0, 
        betastar_III = 0,
        fstar_III = 10**(-2.5),
        Mc_III = 1e7,
        dlog10epsstardz_III = 0.0,
        fesc7_III = 10**(-1.35),
        alphaesc_III = -0.3,
        L40_xray_III = 3.0,
        alpha_xray_III = -1.0,        
        USE_LW_FEEDBACK = False,
        A_LW = 2.0,
        beta_LW = 0.6,
        A_vcb = 1.0,
        beta_vcb = 1.8,
        )



"Class to store the quantities needed in the LIM computation and analysis, define in the input list the ones that you want to vary while the others are fiducial"
class run_oLIMpus:

    def __init__(self, LINE, LINE_MODEL = 'Yang24', _R = 2., shot_noise= False, quadratic_lognormal=True, sigma_LMh = 0., astromodel=0, ZMIN = 5., RSD_MODE = 0, \
        alphastar = 0.5,
        betastar = -0.5,
        epsstar = 0.1,
        Mc = 3e11,
        Mturn_fixed = None,
        dlog10epsstardz = 0.0,):

        self.UP = User_Parameters(
            precisionboost= 1.0, 
            FLAG_FORCE_LINEAR_CF= 0, 
            MIN_R_NONLINEAR= 0.5, 
            MAX_R_NONLINEAR= 200.0,
            FLAG_DO_DENS_NL= False, 
            FLAG_WF_ITERATIVE= True
            )

        self.CP, ClassyC, self.zeus_corr, self.HMFcl =  cosmo_wrapper(self.UP, Cosmo_Parameters_Input(**CosmoParams_input_fid))

        AstroParams_input = copy(AstroParams_input_fid)
        AstroParams_input['astromodel'] = astromodel
        AstroParams_input['quadratic_SFRD_lognormal'] = quadratic_lognormal
        AstroParams_input['alphastar'] = alphastar
        AstroParams_input['betastar'] = betastar
        AstroParams_input['epsstar'] = epsstar
        AstroParams_input['Mturn_fixed'] = Mturn_fixed
        AstroParams_input['Mc'] = Mc
        AstroParams_input['dlog10epsstardz'] = dlog10epsstardz
        self.AP = zeus21.Astro_Parameters(self.UP, self.CP, **AstroParams_input)

        LineParams_Input_val = LineParams_Input(
            LINE = LINE, # which line
            LINE_MODEL = LINE_MODEL, # model of the line luminosity
            OBSERVABLE_LIM = 'Inu', # observe intensity in Jy/sr or mK
            _R = _R, # resolution for smoothing
            sigma_LMh = sigma_LMh, # stochasticity in the L-SFR relation
            shot_noise = shot_noise, # add shot noise to the power spectrum
            Eulerian = True, # Eulerian or Lagrangian space, MOVE TO USER PARAMS
            quadratic_lognormal = quadratic_lognormal # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
        )

        self.LP = Line_Parameters(LineParams_Input_val,self.UP)
        
        self.LIM_coeff = get_LIM_coefficients(self.LP, self.AP, self.CP, self.HMFcl, self.UP, ZMIN)

        self.LIM_corr = Correlations_LIM(self.LP, self.CP, ClassyC)

        self.LIM_pk = Power_Spectra_LIM(self.LIM_corr, self.LIM_coeff, self.LP, self.CP, self.UP, RSD_MODE)

        self.zeus_coeff = get_T21_coefficients(self.UP, self.CP, ClassyC, self.AP, self.HMFcl, ZMIN)

        self.zeus_pk = Power_Spectra(self.UP, self.CP, self.AP, ClassyC, self.zeus_corr, self.zeus_coeff, RSD_MODE)