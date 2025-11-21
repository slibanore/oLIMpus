from study_anticorr import run_LIM, get_reio_field, r_cross, Pearson, alphastar_val, betastar_val, fesc_val, epsstar_val, LX_val
from oLIMpus import CoevalBox_LIM_analytical, CoevalBox_T21reionization 
from oLIMpus import analysis as a 
import numpy as np 
import os
from tqdm import tqdm
import pickle

ZMIN = 5.
RSD_MODE = 1

# user parameters
UP = a.User_Parameters(
            precisionboost= 1., 
            FLAG_FORCE_LINEAR_CF= 0, 
            MIN_R_NONLINEAR= 0.5, 
            MAX_R_NONLINEAR= 200.0,
            FLAG_DO_DENS_NL= False, 
            FLAG_WF_ITERATIVE= True,
            )

# cosmological parameters
CP_fid, ClassyC_fid, zeus_corr_fid, HMFcl_fid = a.cosmo_wrapper(UP, a.Cosmo_Parameters_Input(**a.CosmoParams_input_fid))

# astro parameters fiducial
AstroParams_input_fid_use = dict(
    
        astromodel = 0,
        accretion_model = 0,
        alphastar = 0.5,
        betastar = -0.5,
        epsstar = 0.1,
        Mc = 3e11,
        Mturn_fixed = None,
        dlog10epsstardz = 0.0,
        quadratic_SFRD_lognormal = True, 
    
        fesc10 = 0.1, # !!! 
        alphaesc = 0., 
        L40_xray = 1e41/1e40,
        E0_xray = 500.,
        alpha_xray = -1.0,
        Emax_xray_norm= 2000,

        Nalpha_lyA_II = 9690,
        Nalpha_lyA_III = 17900,

        FLAG_MTURN_SHARP= False,

        C0dust = 4.43,
        C1dust = 1.99,
        sigmaUV=0.5,

        USE_POPIII = False,
        USE_LW_FEEDBACK=False
        )


path = './run_random_parameters/'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

def run_many(Nruns,model,Lbox,with_shotnoise=True,Nbox=None, _R = None,Rmin_bubbles=0.05,compute_mass_weighted_xHII=False,compute_include_partlion=True,compute_partial_and_massweighted=False, extra_label='',seed_input=None,vary_cosmology=False):

    for n in range(Nruns):

        generate_maps_random(n, model,Lbox,Nbox,with_shotnoise,_R,Rmin_bubbles,compute_mass_weighted_xHII,compute_include_partlion,compute_partial_and_massweighted, extra_label,seed_input,vary_cosmology=vary_cosmology)

    return 

def import_random_model(nid, model,Lbox,with_shotnoise=True,Nbox = None, _R=None,include_partlion=True,vary_cosmology=False):

    save_path = path + str(nid) + '_random_' + model + '_' + str(Lbox) + '_' + str(Nbox) 

    if not with_shotnoise:
        save_path += '_noSN'
    if _R != None:
        save_path += '_' + str(_R)
    if vary_cosmology:
        save_path += '_cosmovar_'
    if not include_partlion:
        save_path += '_fullion' 

    save_path += '.pkl'

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No saved output at {save_path}")

    with open(save_path, 'rb') as f:
        saved = pickle.load(f)

    inputs_saved = saved['inputs']
    inputs_current = {
        'with_shotnoise': with_shotnoise
    }

    # Compare inputs
    for key in inputs_current:
        val_current = inputs_current[key]
        val_saved = inputs_saved[key]

        if isinstance(val_current, (list, np.ndarray)):
            # Handle arrays of floats robustly
            if not np.allclose(val_current, val_saved, rtol=1e-6, atol=1e-10):
                raise ValueError(f"Mismatch in input parameter '{key}'")
        else:
            if val_current != val_saved:
                raise ValueError(f"Mismatch in input parameter '{key}'")

    print(f"Successfully loaded model output from {save_path}")
    outputs = saved['outputs']

    return outputs


def generate_maps_random(nid, model,Lbox,Nbox, with_shotnoise=True, _R = None,Rmin_bubbles=0.05,compute_mass_weighted_xHII=False,compute_include_partlion=False,compute_partial_and_massweighted=True, extra_label='',seed_input=None,save_maps =False, vary_cosmology = False):

    if model == 'OIII':
        LP_input = a.LineParams_Input(
            LINE = 'OIII', # which line
            LINE_MODEL = 'Yang24', # model of the line luminosity
            OBSERVABLE_LIM = 'Inu', # observe intensity in Jy/sr or mK
            _R = 1. if _R == None else _R, # resolution for smoothing
            sigma_LMh = 0., # stochasticity in the L-SFR relation
            shot_noise = with_shotnoise, # add shot noise to the power spectrum
            quadratic_lognormal = True, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
            line_dict = None
        )

    elif model == 'Ha':
        LP_input = a.LineParams_Input(
            LINE = 'Ha', # which line
            LINE_MODEL = 'Yang24', # model of the line luminosity
            OBSERVABLE_LIM = 'Inu', # observe intensity in Jy/sr or mK
            _R =  1. if _R == None else _R, # resolution for smoothing
            sigma_LMh = 0., # stochasticity in the L-SFR relation
            shot_noise = with_shotnoise, # add shot noise to the power spectrum
            quadratic_lognormal = True, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
            line_dict = None
        )

    elif model == 'CII':
        LP_input = a.LineParams_Input(
            LINE = 'CII', # which line
            LINE_MODEL = 'Lagache18', # model of the line luminosity
            OBSERVABLE_LIM = 'Inu', # observe intensity in Jy/sr or mK
            _R =  1. if _R == None else _R, # resolution for smoothing
            sigma_LMh = 0., # stochasticity in the L-SFR relation
            shot_noise = with_shotnoise, # add shot noise to the power spectrum
            quadratic_lognormal = True, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
            line_dict = None
        )

    elif model == 'CO21':
        LP_input = a.LineParams_Input(
            LINE = 'CO21', # which line
            LINE_MODEL = 'Li16', # model of the line luminosity
            OBSERVABLE_LIM = 'Tnu', # observe intensity in Jy/sr or mK
            _R =  1. if _R == None else _R, # resolution for smoothing
            sigma_LMh = 0., # stochasticity in the L-SFR relation
            shot_noise = with_shotnoise, # add shot noise to the power spectrum
            quadratic_lognormal = True, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
            line_dict = None
        )
        
    elif model == 'SFRD' or model == 'SFRDxH' or model == 'SFRDT21nb':
        LP_input = a.LineParams_Input(
            LINE = 'OIII', # which line
            LINE_MODEL = 'powerlaw', # model of the line luminosity
            OBSERVABLE_LIM = 'Inu', # observe intensity in Jy/sr or mK
            _R =  1. if _R == None else _R, # resolution for smoothing
            sigma_LMh = 0., # stochasticity in the L-SFR relation
            shot_noise = with_shotnoise, # add shot noise to the power spectrum
            quadratic_lognormal = True, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
            line_dict = {'alpha_SFR':1.}
        )

    LP = a.Line_Parameters(LP_input,UP)


    alphastar, betastar, epsstar, fesc, LX = extract_parameters()
    if vary_cosmology:
        CP, ClassyC, HMFcl, zeus_corr = extract_cosmo_parameters()
    else:
        CP=CP_fid
        ClassyC=ClassyC_fid
        HMFcl=HMFcl_fid 
        zeus_corr=zeus_corr_fid

    AP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_corr, zeus_pk = run_LIM(alphastar=alphastar,betastar=betastar,epsstar=epsstar,Mturn_fixed=None,Mc=None,fesc=fesc,LX=LX,LP=LP,CP=CP, ClassyC=ClassyC, HMFcl=HMFcl, zeus_corr=zeus_corr)

    zvals = zeus_coeff.zintegral 

    xHv = np.zeros((len(zvals)))
    T21 = np.zeros((len(zvals)))
    p = np.zeros((len(zvals)))
    r = []
    k_cross = [] 
    pk_cross = [] 
    pk_auto_21 = [] 
    use_pk_auto_line = []

    k_bins = np.logspace(np.log10(2*np.pi/Lbox),np.log10(2*np.pi/Lbox*Nbox),10) # k array 

    seed = np.random.randint(0, 2**32) if seed_input == None else seed_input

    save_path = path +  'random_maps_L' + str(Lbox) +'_N' + str(Nbox) +  '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_parameters_path = save_path + 'par_values.dat' 
    if not os.path.exists(save_parameters_path):
        with open(save_parameters_path, "w") as f:  # "w" to create new file
            row_str = "nid\tepsstar\talphastar\tbetastar\tfesc\tLX"
            f.write(f"{row_str}\n")

    with open(save_parameters_path, "a") as f:
        f.write(f"{nid:d}\t")
        f.write(f"{epsstar:.3f}\t")
        f.write(f"{alphastar:.3f}\t")
        f.write(f"{betastar:.3f}\t")
        f.write(f"{fesc:.3f}\t")
        f.write(f"{LX:.3f}\n")
 
    reionization_map_partial, ion_frac_withpartial = get_reio_field(zvals,zeus_coeff, zeus_corr, AP, CP, ClassyC, HMFcl, Lbox, Nbox, Rmin_bubbles, seed=seed, compute_mass_weighted_xHII=compute_mass_weighted_xHII,compute_include_partlion=compute_include_partlion,compute_partial_and_massweighted=compute_partial_and_massweighted)


    for zv in tqdm(range(len(zvals))):

        box_line_all = CoevalBox_LIM_analytical(LIM_coeff,LIM_corr,LIM_pk,LP,zvals[zv],LP._R,Lbox,Nbox, RSD=RSD_MODE, get_density_box=True,seed=seed)

        if with_shotnoise:
            box_line = box_line_all.Inu_box_smooth
        else:
            box_line = box_line_all.Inu_box_noiseless_smooth


        box_T21 = CoevalBox_T21reionization(zeus_coeff, zeus_pk, zvals[zv], reionization_map_partial[zv], ion_frac_withpartial[zv], Lbox, Nbox, seed, MAP_T21_FULL = True, input_Resolution=_R)

        xHv[zv] = box_T21.xH_avg_map
        T21[zv] = np.mean(box_T21.T21_map)

        SNlabel = '_noSN' if not with_shotnoise else ''
        save_path_line = save_path + str(nid) + '_random_' + LP.LINE + SNlabel + extra_label + '.dat'
        save_path_T21 = save_path +  str(nid) + '_random_T21' + SNlabel + extra_label + '.dat'
        save_path_xH = save_path + str(nid) + '_random_' + 'xH'+ SNlabel +extra_label + '.dat'

        if save_maps:

            with open(save_path_line, "a") as f_line, open(save_path_xH, "a") as f_xH, open(save_path_T21, "a") as f_21:
                for x in range(Nbox):
                    box_T21.T21_map[0][x][np.isnan(box_T21.T21_map[0][x])] = 0.
                    row_str = " ".join(f"{val:.6e}" for val in box_line[0][x])
                    f_line.write(f"{zvals[zv]:.3f} {row_str}\n")
                    row_str = " ".join(f"{val:.6e}" for val in box_T21.xH_box[0][x])
                    f_xH.write(f"{zvals[zv]:.3f} {row_str}\n")
                    row_str = " ".join(f"{val:.6e}" for val in box_T21.T21_map[0][x])
                    f_21.write(f"{zvals[zv]:.3f} {row_str}\n")

        temp_v = r_cross(box_line,box_T21.T21_map, Lbox, k_bins, foregrounds=False)
        p[zv] = Pearson(box_line,box_T21.T21_map,False)

        r.append(temp_v[0])
        k_cross.append(temp_v[1]) 
        pk_cross.append(temp_v[2]) 
        pk_auto_21.append(temp_v[3]) 
        use_pk_auto_line.append(temp_v[4])

    # Save inputs and outputs to file
    data_to_save = {
        'inputs': {
            'with_shotnoise': with_shotnoise
        },
        'outputs': {
            'p': p,
            'T21': T21,
            'xHv': xHv,
            'r': r,
            'k_cross': k_cross,
            'pk_cross': pk_cross,
            'pk_auto_21': pk_auto_21,
            'use_pk_auto_line': use_pk_auto_line
        }
    }

    save_path = path + str(nid) + '_random_' + model + '_' + str(Lbox) + '_' + str(Nbox) 
    if not with_shotnoise:
        save_path += '_noSN'
    if _R != None:
        save_path += '_' + str(_R)
    if vary_cosmology:
        save_path += '_cosmovar_'
    if not compute_include_partlion:
        save_path += '_fullion' 
    save_path += extra_label
    save_path += '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    return 

Nsigma = 3.
alphastar_fid = AstroParams_input_fid_use['alphastar']
sigma_alphastar = (alphastar_fid-alphastar_val[0])/Nsigma
betastar_fid = AstroParams_input_fid_use['betastar']
sigma_betastar = (betastar_fid-betastar_val[0])/Nsigma
epsstar_fid = AstroParams_input_fid_use['epsstar']
sigma_epsstar = (epsstar_fid-epsstar_val[0])/Nsigma
fesc_fid = AstroParams_input_fid_use['fesc10']
sigma_fesc = (fesc_fid-fesc_val[0])/Nsigma
LX_fid = np.log10(AstroParams_input_fid_use['L40_xray'] * 1e40)
sigma_LX = (LX_fid-LX_val[0])/Nsigma

def extract_parameters():

    alphastar = np.random.normal(alphastar_fid, sigma_alphastar)
    betastar = np.random.normal(betastar_fid, sigma_betastar)
    epsstar = np.random.normal(epsstar_fid, sigma_epsstar)
    fesc =  np.random.normal(fesc_fid, sigma_fesc)
    LX = np.random.normal(LX_fid, sigma_LX)

    return alphastar, betastar, epsstar, fesc, LX 


def extract_cosmo_parameters():

    # sigma from Tab 1 in 1807.06209
    omegab = np.random.normal(a.CosmoParams_input_fid['omegab'],0.00015)
    omegac = np.random.normal(a.CosmoParams_input_fid['omegac'],0.0012)
    h_fid = np.random.normal(a.CosmoParams_input_fid['h_fid'],0.54/100.)
    As = np.random.normal(a.CosmoParams_input_fid['As'],np.exp(0.014)/1e10)
    ns = np.random.normal(a.CosmoParams_input_fid['ns'],0.0042)
    tau_fid = np.random.normal(a.CosmoParams_input_fid['tau_fid'],0.0074)

    CosmoParams_input = dict(
        omegab= omegab, 
        omegac = omegac, 
        h_fid = h_fid, 
        As = As, 
        ns = ns, 
        tau_fid = tau_fid, 
        HMF_CHOICE= "ST",
        Flag_emulate_21cmfast = False,
        )
    
    CP, ClassyC, zeus_corr, HMFcl = a.cosmo_wrapper(UP, a.Cosmo_Parameters_Input(**CosmoParams_input))


    return CP, ClassyC, HMFcl, zeus_corr