from oLIMpus import analysis as a 
from oLIMpus import CoevalBox_LIM_analytical, CoevalBox_T21reionization, CoevalBox_LIM_analytical
from oLIMpus import get_T21_coefficients, Correlations, Power_Spectra, constants
from oLIMpus import get_reio_field
import numpy as np 
import powerbox as pb
from copy import copy 
import pickle
import os 
from scipy.interpolate import interp1d
from tqdm import tqdm

import matplotlib.pyplot as plt 
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
CP, ClassyC, zeus_corr, HMFcl = a.cosmo_wrapper(UP, a.Cosmo_Parameters_Input(**a.CosmoParams_input_fid))

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


# set to get reionizatin between 5 and 8 when all other parameters are set to their fiducials
epsstar_val = [0.01,0.04,0.3,0.9]#np.logspace(np.log10(0.04),np.log10(0.3),10)
fesc_val = [0.01,0.05,0.25,0.5]#np.logspace(np.log10(0.05),np.log10(0.15),10)
alphastar_val = np.linspace(0.3,0.8,10)
betastar_val = np.linspace(-1,0.,10)
LX_val = [39.5,40.5,41.5,42.5]#np.linspace(40.5,41.5,10)

path = './runs/'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


def import_model(model,which_par,par_vals,Lbox,with_shotnoise=True,Nbox = None, _R=None,include_partlion=True):

    save_path = path + model + '_' + which_par + '_' + str(Lbox) + '_' + str(Nbox) 
    if not with_shotnoise:
        save_path += '_noSN'
    if _R != None:
        save_path += '_' + str(_R)
    if not include_partlion:
        save_path += '_fullion' 

    save_path += '.pkl'

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No saved output at {save_path}")

    with open(save_path, 'rb') as f:
        saved = pickle.load(f)

    inputs_saved = saved['inputs']
    inputs_current = {
        'par_vals': par_vals,
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


def run_and_save_model(model,which_par,par_vals,Lbox,with_shotnoise=True,Nbox=None,save_maps=False, _R = None,Rmin_bubbles=0.05,compute_mass_weighted_xHII=False,compute_include_partlion=False,compute_partial_and_massweighted=True, extra_label='',seed_input=None):

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

    if which_par == 'fstar':
        mm = lambda epsstar: run_LIM(alphastar=None,betastar=None,epsstar=epsstar,Mturn_fixed=None,Mc=None,fesc=None,LX=None,LP=LP)
    elif which_par == 'fesc':
        mm = lambda fesc: run_LIM(alphastar=None,betastar=None,epsstar=None,Mturn_fixed=None,Mc=None,fesc=fesc,LX=None,LP=LP)
    elif which_par == 'alphastar':
        mm = lambda alpha: run_LIM(alphastar=alpha,betastar=None,epsstar=None,Mturn_fixed=None,Mc=None,fesc=None,LX=None,LP=LP)
    elif which_par == 'betastar':
        mm = lambda beta: run_LIM(alphastar=None,betastar=beta,epsstar=None,Mturn_fixed=None,Mc=None,fesc=None,LX=None,LP=LP)
    elif which_par == 'LX':
        mm = lambda LX: run_LIM(alphastar=None,betastar=None,epsstar=None,Mturn_fixed=None,Mc=None,fesc=None,LX=LX,LP=LP)
    elif which_par == 'fiducial':
        par_vals = [0]
        mm = lambda x: run_LIM(alphastar=None,betastar=None,epsstar=None,Mturn_fixed=None,Mc=None,fesc=None,LX=None,LP=LP)

    p = []
    T21 = []
    xHv = []
    r = [] 
    k_cross = [] 
    pk_cross = [] 
    pk_auto_21 = [] 
    use_pk_auto_line = []

    k_bins = np.logspace(np.log10(2*np.pi/Lbox),np.log10(2*np.pi/Lbox*Nbox),10) # k array 

    for i in tqdm(range(len(par_vals))):
        seed = np.random.randint(0, 2**32) if seed_input == None else seed_input
        AP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_corr, zeus_pk = mm(par_vals[i])

        if save_maps:
            zvals = zeus_coeff.zintegral 
        else:   
            zvals = zeus_coeff.zintegral

        xHv.append(np.zeros((len(zvals))))
        T21.append(np.zeros((len(zvals))))
        p.append(np.zeros((len(zvals))))
        r.append([])
        k_cross.append([]) 
        pk_cross.append([]) 
        pk_auto_21.append([]) 
        use_pk_auto_line.append([])

        if save_maps and not os.path.exists(path+  'maps_L' + str(Lbox) +'_N' + str(Nbox) +  '/'):
            os.makedirs(path +  'maps_L' + str(Lbox) +'_N' + str(Nbox) +  '/')

        reionization_map_partial, ion_frac_withpartial = get_reio_field(zvals,zeus_coeff, zeus_corr, AP, CP, ClassyC, HMFcl, Lbox, Nbox, Rmin_bubbles, seed=seed, compute_mass_weighted_xHII=compute_mass_weighted_xHII,compute_include_partlion=compute_include_partlion,compute_partial_and_massweighted=compute_partial_and_massweighted)

        for zv in tqdm(range(len(zvals))):

            box_line_all = CoevalBox_LIM_analytical(LIM_coeff,LIM_corr,LIM_pk,LP,zvals[zv],LP._R,Lbox,Nbox, RSD=RSD_MODE, get_density_box=True,seed=seed,)

            if with_shotnoise:
                box_line = box_line_all.Inu_box_smooth
            else:
                box_line = box_line_all.Inu_box_noiseless_smooth

            box_T21 = CoevalBox_T21reionization(zeus_coeff, zeus_pk, zvals[zv], reionization_map_partial[zv], ion_frac_withpartial[zv], Lbox, Nbox, seed, MAP_T21_FULL = True,input_Resolution=_R)

            xHv[i][zv] = box_T21.xH_avg_map
            T21[i][zv] = np.mean(box_T21.T21_map)

            if which_par == 'fiducial' and save_maps:

                SNlabel = '_noSN' if not with_shotnoise else ''
                if model != 'SFRD':
                    save_path_line = path + 'maps_L' + str(Lbox) + '_N' + str(Nbox) + '/' + LP.LINE + SNlabel + extra_label + '.dat'
                else:
                    save_path_line = path + 'maps_L' + str(Lbox) + '_N' + str(Nbox) + '/SFRD' + SNlabel + extra_label + '.dat'
                    
                save_path_T21 = path +  'maps_L' + str(Lbox) + '_N' + str(Nbox) + '/' + 'T21' + SNlabel + extra_label + '.dat'
                save_path_xH = path +  'maps_L' + str(Lbox) + '_N' + str(Nbox) + '/' + 'xH'+ SNlabel +extra_label + '.dat'
                save_path_Pearson = path +  'maps_L' + str(Lbox) + '_N' + str(Nbox) + '/' + 'P'+ SNlabel +extra_label + '.dat'

                if model == 'SFRD':
                    save_path_density =  path + 'maps_L' + str(Lbox) + '_N' + str(Nbox) + '/delta_'+ SNlabel + extra_label + '.dat'

                    box_delta = box_line_all.density_box_smooth

                    with open(save_path_density, "a") as f_den:
                        for x in range(Nbox):
                                row_str = " ".join(f"{val:.6e}" for val in box_delta[0][x])
                                f_den.write(f"{zvals[zv]:.3f} {row_str}\n")

                box_Pearson = (box_T21.T21_map-T21[0][zv])*(box_line-np.mean(box_line)) #/ np.sqrt(np.sum((box_T21.T21_map[0]-T21[0][zv])**2)*np.sum((box_line[0]-np.mean(box_line[0]))**2))

                with open(save_path_line, "a") as f_line, open(save_path_xH, "a") as f_xH, open(save_path_T21, "a") as f_21, open(save_path_Pearson, "a") as f_P:
                    for x in range(Nbox):
                            box_T21.T21_map[0][x][np.isnan(box_T21.T21_map[0][x])] = 0.
                            row_str = " ".join(f"{val:.6e}" for val in box_line[0][x])
                            f_line.write(f"{zvals[zv]:.3f} {row_str}\n")
                            row_str = " ".join(f"{val:.6e}" for val in box_T21.xH_box[0][x])
                            f_xH.write(f"{zvals[zv]:.3f} {row_str}\n")
                            row_str = " ".join(f"{val:.6e}" for val in box_T21.T21_map[0][x])
                            f_21.write(f"{zvals[zv]:.3f} {row_str}\n")
                            row_str = " ".join(f"{val:.6e}" for val in box_Pearson[x])
                            f_P.write(f"{zvals[zv]:.3f} {row_str}\n")

            if model == 'SFRDxH':
                temp_v = r_cross(box_line,box_T21.xH_box, Lbox, k_bins, foregrounds=False)
                p[i][zv] = Pearson_unnorm(box_line,box_T21.xH_box,False)
                fig,ax = plt.subplots(1,2)
                ax[0].set_title(r'$z=%g$'%zvals[zv])
                ax[1].set_title(r'$z=%g$'%zvals[zv])
                # im = ax[0].imshow(box_line[0])
                # im = ax[1].imshow(box_T21.xH_box[0])
                # plt.show()
            elif model == 'SFRDT21nb':
                temp_v = r_cross(box_line,box_T21.T21_map_only, Lbox, k_bins, foregrounds=False)
                p[i][zv] = Pearson(box_line,box_T21.T21_map_only,False)
            else:
                temp_v = r_cross(box_line,box_T21.T21_map, Lbox, k_bins, foregrounds=False)
                p[i][zv] = Pearson(box_line,box_T21.T21_map,False)

            r[i].append(temp_v[0])
            k_cross[i].append(temp_v[1]) 
            pk_cross[i].append(temp_v[2]) 
            pk_auto_21[i].append(temp_v[3]) 
            use_pk_auto_line[i].append(temp_v[4])

    # Save inputs and outputs to file
    data_to_save = {
        'inputs': {
            'par_vals': par_vals,
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

    save_path = path + model + '_' + which_par + '_' + str(Lbox) + '_' + str(Nbox) 
    if not with_shotnoise:
        save_path += '_noSN'
    if _R != None:
        save_path += '_' + str(_R)
    if not compute_include_partlion and not compute_partial_and_massweighted:
        save_path += '_fullion' 
    save_path += '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    return


def run_LIM(alphastar,betastar,epsstar,Mturn_fixed,Mc,fesc,LX,LP):

    AP = change_astro(alphastar,betastar,epsstar,Mturn_fixed,Mc,fesc,LX)

    # LIM
    LIM_coeff = a.get_LIM_coefficients(LP, AP, CP, HMFcl, UP, ZMIN)

    LIM_corr = a.Correlations_LIM(LP, CP, ClassyC)

    LIM_pk = a.Power_Spectra_LIM(LIM_corr, LIM_coeff, LP, CP, UP, RSD_MODE)

    # power spectra 
    zeus_coeff = get_T21_coefficients(UP, CP, ClassyC, AP, HMFcl, ZMIN)

    zeus_corr = Correlations(UP, CP, ClassyC)

    zeus_pk = Power_Spectra(UP, CP, AP, ClassyC, zeus_corr, zeus_coeff, RSD_MODE)

    return AP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_corr, zeus_pk


def change_astro(alphastar,betastar,epsstar,Mturn_fixed,Mc,fesc,LX):

    AstroParams_input = copy(AstroParams_input_fid_use)
    if alphastar is not None:
        AstroParams_input['alphastar'] = alphastar
    if betastar is not None:
        AstroParams_input['betastar'] = betastar
    if epsstar is not None:
        AstroParams_input['epsstar'] = epsstar
    if Mturn_fixed is not None:
        AstroParams_input['Mturn_fixed'] = Mturn_fixed
    if Mc is not None:
        AstroParams_input['Mc'] = Mc
    if fesc is not None:
        AstroParams_input['fesc10'] = fesc
    if LX is not None:
        AstroParams_input['L40_xray'] = 10**LX/1e40
        
    AP = a.zeus21.Astro_Parameters(UP, CP, **AstroParams_input)
    
    return AP


def Pearson(box_LIM, box_T21, foregrounds):

    cross_TLIM1 = np.corrcoef((box_T21.flatten()), box_LIM.flatten())[0, 1]
    if foregrounds:
        print('Foregrounds not yet implemented')

    return cross_TLIM1

def Pearson_unnorm(box_LIM, box_T21, foregrounds):

    cross_TLIM1 = np.cov((box_T21.flatten()), box_LIM.flatten())[0,1]
    if foregrounds:
        print('Foregrounds not yet implemented')

    return cross_TLIM1




def r_cross(box_LIM, box_T21, Lbox, k_bins, foregrounds):

    if foregrounds:
        print('Foregrounds not yet implemented')

    use_pk_cross, k_cross = pb.get_power(
    deltax = box_T21,
    boxlength= Lbox,
    deltax2 = box_LIM,
    bins = k_bins
    )

    use_pk_auto_21, _ = pb.get_power(
    deltax = box_T21,
    boxlength= Lbox,
    bins = k_bins
    )

    use_pk_auto_line, _ = pb.get_power(
    boxlength= Lbox,
    deltax = box_LIM,
    bins = k_bins
    )

    r = use_pk_cross/np.sqrt(use_pk_auto_line*use_pk_auto_21)

    return r, k_cross, use_pk_cross, use_pk_auto_21, use_pk_auto_line


