"""
Author: Emilie Thelie (emilie.thelie@austin.utexas.edu), adapted codes from Sarah Libanore
UT Austin - August 2025

To run this file:
python cross_n_bubbles.py 0 OIII 250 250 1 0 1 0.075 41

Parameters are:
OVERWRITE_FILES which_field Lbox Nbox seed shot_noise quadratic_SFRD_lognormal fesc LX
(fesc and LX default are 0.075 41, which we call with -1)
"""
import numpy as np
from scipy import interpolate, integrate

import oLIMpus as olim
import oLIMpus.analysis as olima
from oLIMpus.zeus21_local.zeus21.maps import reionization_maps as reio
from copy import copy 
import powerbox as pb
import mcfit

import os
import sys
import gc
import time



##### run and save models
def change_astro(UP,CP,AstroParams_input_fid_use,alphastar,betastar,epsstar,Mturn_fixed,Mc,fesc,LX):
    AstroParams_input = copy(AstroParams_input_fid_use)
    if alphastar is not None:
        AstroParams_input["alphastar"] = alphastar
    if betastar is not None:
        AstroParams_input["betastar"] = betastar
    if epsstar is not None:
        AstroParams_input["epsstar"] = epsstar
    if Mturn_fixed is not None:
        AstroParams_input["Mturn_fixed"] = Mturn_fixed
    if Mc is not None:
        AstroParams_input["Mc"] = Mc
    if fesc is not None:
        AstroParams_input["fesc10"] = fesc
    if LX is not None:
        AstroParams_input["L40_xray"] = 10**LX/1e40
        
    AP = olim.zeus21.Astro_Parameters(UP, CP, **AstroParams_input)
    
    return AP

def run_LIM(UP,CP,ClassyC,AP,LP,HMFcl,ZMIN,RSD_MODE,alphastar=None,betastar=None,epsstar=None,Mturn_fixed=None,Mc=None,fesc=None,LX=None):
    AP = change_astro(UP,CP,AP,alphastar,betastar,epsstar,Mturn_fixed,Mc,fesc,LX)
    
    # LIM
    LIM_coeff = olim.get_LIM_coefficients(LP, AP, CP, HMFcl, UP, ZMIN)
    LIM_corr = olim.Correlations_LIM(LP, CP, ClassyC)
    LIM_pk = olim.Power_Spectra_LIM(LIM_corr, LIM_coeff, LP, CP, UP, RSD_MODE)

    # power spectra 
    zeus_coeff = olim.get_T21_coefficients(UP, CP, ClassyC, AP, HMFcl, ZMIN)
    zeus_corr = olim.Correlations(UP, CP, ClassyC)
    zeus_pk = olim.Power_Spectra(UP, CP, AP, ClassyC, zeus_corr, zeus_coeff, RSD_MODE)

    return AP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_corr, zeus_pk



##### get boxes
def get_boxes(LP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_pk, zvals, 
              reionization_map_partial, ion_frac_withpartial, Lbox, Nbox, seed, RSD_MODE):
    SFRD_box, T21_box = np.zeros((len(zvals),Nbox,Nbox,Nbox)), np.zeros((len(zvals),Nbox,Nbox,Nbox))
    T21_avg = np.zeros((len(zvals)))
    for i_z, zval in enumerate(zvals):
        box_line_all = olim.CoevalBox_LIM_analytical(LIM_coeff, LIM_corr, LIM_pk, LP, zval, LP._R, Lbox, Nbox, 
                                                     RSD=RSD_MODE, get_density_box=True, seed=seed)
        box_T21 = olim.CoevalBox_T21reionization(zeus_coeff, zeus_pk, zval, 
                                                 reionization_map_partial[i_z], ion_frac_withpartial, Lbox, Nbox, seed, 
                                                 MAP_T21_FULL = True, input_Resolution=1)
        # boxes
        SFRD_box[i_z] = box_line_all.Inu_box_smooth
        T21_box[i_z] = box_T21.T21_map
        # averages
        T21_avg[i_z] = np.mean(T21_box[i_z])
        # delete class instance
        delete_class_attributes(box_line_all) ; del box_line_all
        delete_class_attributes(box_T21) ; del box_T21
        gc.collect()

    return SFRD_box, T21_box, T21_avg



##### get correlations
def my_compute_pk(field, Lbox, kmin=None, kmax=None, Nbins=25, KBINS_IN_LOG=False):
    # not used here but a longer version of a calculation of the Pk from boxes
    Ncell = field.shape[0] 
    Dim = len(field.shape)
    res_cell = Lbox / Ncell
    if KBINS_IN_LOG:
        if kmin is None:
            kmin = np.log10(2*np.pi/Lbox)
        else:
            kmin = np.log10(kmin)
        if kmax is None:
            kmax = np.log10(2*np.pi/res_cell)
        else:
            kmax = np.log10(kmax)
        bk = np.logspace(kmin,kmax,Nbins)
    else:
        if kmin is None:
            kmin = 2*np.pi/Lbox
        if kmax is None:
            kmax = 2*np.pi/res_cell
        bk = np.linspace(kmin,kmax,Nbins)

    # get |k|
    kfreq = np.fft.fftfreq(Ncell,res_cell) * 2*np.pi
    kvec = np.array(np.meshgrid(*[kfreq]*Dim, sparse=False, indexing="ij"))
    kfreq2 = np.sqrt(np.sum(kvec**2,axis=0))
    # hist of k
    nk, bin_k = np.histogram(kfreq2,bk)  
    # fft of the field
    ffs = np.fft.fftn(field)
    ffs2 = ffs*np.conjugate(ffs)
    # compute pk
    pk, bin_k = np.histogram(kfreq2,bin_k,weights=ffs2)
    pk = pk/(1.0*nk+(nk==0)) / Ncell**Dim * res_cell**Dim
    
    bin_k = (bin_k[:-1]+bin_k[1:])*0.5
    return pk.real, bin_k

def compute_pk(box1, Lbox, k_bins, box2=None):
    if box2 is None:
        pk, k_bins, var = pb.get_power(deltax=box1-np.mean(box1), boxlength=Lbox, bins=k_bins, get_variance=True)
    else:
        pk, k_bins, var = pb.get_power(deltax=box1-np.mean(box1), boxlength=Lbox, deltax2=box2-np.mean(box2), bins=k_bins, get_variance=True)
    return k_bins, pk, var
    
def r_cross(pk1, pk2, pk_cross):
    return pk_cross/(np.sqrt(pk1*pk2)+1e-10)

def Pearson(box1, box2):
    p = np.corrcoef((box1.flatten()), box2.flatten())[0, 1]
    return p

# two point correlation : these functions uses the power spectrum (by doing the FT or using mcfit), otherwise the pylians library directly extracts xi from boxes
def compute_xi_FT(klist, Pk, Rmin=0.5, Rmax=2e3, Nbins=25, RBINS_IN_LOG=False):
    pk_interp = interpolate.interp1d(klist,Pk,bounds_error=False,fill_value=0.)
    if RBINS_IN_LOG:
        rlist = np.logspace(np.log10(Rmin),np.log10(Rmax),Nbins)
    else:
        rlist = np.linspace(Rmin,Rmax,Nbins)

    xi = np.zeros(len(rlist))
    for i, r in enumerate(rlist):
        def corrfunc_integrand(k):
            return k**2 * pk_interp(k) * np.sin(k*r)/(k*r)
        xi[i] = 1/(2*np.pi**2) * integrate.quad(corrfunc_integrand,0,np.inf)[0]
    return rlist, xi

def compute_xi_mcfit(klist, Pk, Rmin=0.5, Rmax=2e3, Nbins=25): 
    # generate mcfit k values
    r_sample = np.logspace(np.log10(Rmin), np.log10(Rmax), Nbins)
    k_sample, _ = mcfit.xi2P(r_sample, l=0, lowring=True)(0*r_sample, extrap=False)

    # get the power spectrum at mcfit k values
    pk_interp =  interpolate.interp1d(klist, Pk, bounds_error=False, fill_value=0.)
    pk_sample = np.array([pk_interp(k) for k in k_sample])

    # compute xi(r)
    r, xi = mcfit.P2xi(k_sample,lowring=True)(pk_sample)

    return r, xi

def get_corr(SFRD_box, xHII_box, T21_box, Lbox, k_bins, zvals, WHICH_COMPUTE_XI="mcfit", Rmin_xi=0.5, Rmax_xi=250, N_rbins=25):
    k_bins_pb, r_bins = np.zeros(len(k_bins)), np.zeros(N_rbins)
    
    pk_SFRD, pk_xHII, pk_xHI, pk_T21 = np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1))
    pkvar_SFRD, pkvar_xHII, pkvar_xHI, pkvar_T21 = np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1))
    pk_SFRDxHII, pk_SFRDxHI, pk_SFRDT21  = np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1))
    pkvar_SFRDxHII, pkvar_SFRDxHI, pkvar_SFRDT21 = np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1))
    r_SFRDxHII, r_SFRDxHI, r_SFRDT21 = np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1)), np.zeros((len(zvals),len(k_bins)-1))
    
    xi_SFRD, xi_xHII, xi_xHI, xi_T21 = np.zeros((len(zvals),N_rbins)), np.zeros((len(zvals),N_rbins)), np.zeros((len(zvals),N_rbins)), np.zeros((len(zvals),N_rbins))
    xi_SFRDxHII, xi_SFRDxHI, xi_SFRDT21 = np.zeros((len(zvals),N_rbins)), np.zeros((len(zvals),N_rbins)), np.zeros((len(zvals),N_rbins))
    
    pearson_SFRDxHII, pearson_SFRDxHI, pearson_SFRDT21 = np.zeros((len(zvals))), np.zeros((len(zvals))), np.zeros((len(zvals)))
    
    for i_z, zval in enumerate(zvals):
        ### power spectrum statistics
        k_bins_pb, pk_SFRD[i_z], pkvar_SFRD[i_z] = compute_pk(SFRD_box[i_z], Lbox, k_bins)
        # SFRD x xHII
        _, pk_xHII[i_z], pkvar_xHII[i_z] = compute_pk(xHII_box[i_z], Lbox, k_bins)
        _, pk_SFRDxHII[i_z], pkvar_SFRDxHII[i_z] = compute_pk(SFRD_box[i_z], Lbox, k_bins, box2=xHII_box[i_z])
        r_SFRDxHII[i_z] = r_cross(pk_SFRD[i_z], pk_xHII[i_z], pk_SFRDxHII[i_z])
        pearson_SFRDxHII[i_z] = Pearson(SFRD_box[i_z], xHII_box[i_z])
        # SFRD x xHI
        _, pk_xHI[i_z], pkvar_xHI[i_z] = compute_pk(1-xHII_box[i_z], Lbox, k_bins)
        _, pk_SFRDxHI[i_z], pkvar_SFRDxHI[i_z] = compute_pk(SFRD_box[i_z], Lbox, k_bins, box2=1-xHII_box[i_z])
        r_SFRDxHI[i_z] = r_cross(pk_SFRD[i_z], pk_xHI[i_z], pk_SFRDxHI[i_z])
        pearson_SFRDxHI[i_z] = Pearson(SFRD_box[i_z], 1-xHII_box[i_z])
        # SFRD x T21
        _, pk_T21[i_z], pkvar_T21[i_z] = compute_pk(T21_box[i_z], Lbox, k_bins)
        _, pk_SFRDT21[i_z], pkvar_SFRDT21[i_z] = compute_pk(SFRD_box[i_z], Lbox, k_bins, box2=T21_box[i_z])
        r_SFRDT21[i_z] = r_cross(pk_SFRD[i_z], pk_T21[i_z], pk_SFRDT21[i_z])
        pearson_SFRDT21[i_z] = Pearson(SFRD_box[i_z], T21_box[i_z])

        ### xi
        if WHICH_COMPUTE_XI=="FT": # using the FT function increases the computation time
            r_bins, xi_SFRD[i_z] = compute_xi_FT(k_bins_pb, pk_SFRD[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            # SFRD x xHII
            _, xi_xHII[i_z] = compute_xi_FT(k_bins_pb, pk_xHII[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            _, xi_SFRDxHII[i_z] = compute_xi_FT(k_bins_pb, pk_SFRDxHII[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            # SFRD x xHI
            _, xi_xHI[i_z] = compute_xi_FT(k_bins_pb, pk_xHI[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            _, xi_SFRDxHI[i_z] = compute_xi_FT(k_bins_pb, pk_SFRDxHI[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            # SFRD x T21
            _, xi_T21[i_z] = compute_xi_FT(k_bins_pb, pk_T21[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            _, xi_SFRDT21[i_z] = compute_xi_FT(k_bins_pb, pk_SFRDT21[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
        elif WHICH_COMPUTE_XI=="mcfit":
            r_bins, xi_SFRD[i_z] = compute_xi_mcfit(k_bins_pb, pk_SFRD[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            # SFRD x xHII
            _, xi_xHII[i_z] = compute_xi_mcfit(k_bins_pb, pk_xHII[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            _, xi_SFRDxHII[i_z] = compute_xi_mcfit(k_bins_pb, pk_SFRDxHII[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            # SFRD x xHI
            _, xi_xHI[i_z] = compute_xi_mcfit(k_bins_pb, pk_xHI[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            _, xi_SFRDxHI[i_z] = compute_xi_mcfit(k_bins_pb, pk_SFRDxHI[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            # SFRD x T21
            _, xi_T21[i_z] = compute_xi_mcfit(k_bins_pb, pk_T21[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)
            _, xi_SFRDT21[i_z] = compute_xi_mcfit(k_bins_pb, pk_SFRDT21[i_z],Rmin=Rmin_xi, Rmax=Rmax_xi, Nbins=N_rbins)

    pk_auto = [pk_SFRD, pk_xHII, pk_xHI, pk_T21]
    pkvar_auto = [pkvar_SFRD, pkvar_xHII, pkvar_xHI, pkvar_T21]
    pk_cross = [pk_SFRDxHII, pk_SFRDxHI, pk_SFRDT21]
    pkvar_cross = [pkvar_SFRDxHII, pkvar_SFRDxHI, pkvar_SFRDT21]
    r = [r_SFRDxHII, r_SFRDxHI, r_SFRDT21]
    pearson = [pearson_SFRDxHII, pearson_SFRDxHI, pearson_SFRDT21]
    xi_auto = [xi_SFRD, xi_xHII, xi_xHI, xi_T21]
    xi_cross = [xi_SFRDxHII, xi_SFRDxHI, xi_SFRDT21]

    return k_bins_pb, r_bins, pk_auto, pk_cross, r, pearson, xi_auto, xi_cross, pkvar_auto, pkvar_cross



##### utilities
def delete_class_attributes(class_instance): # delete all attributes of the class instance
    for attr in list(class_instance.__dict__):    
        delattr(class_instance, attr)
    gc.collect()




if __name__=="__main__":
    global_start_time = time.time()
    #################################
    ##### Initialisation & parameters
    #################################
    print("\n***** Initialisation & parameters *****")
    start_time = time.time()
    OVERWRITE_FILES = bool(int(sys.argv[1]))
    which_field = sys.argv[2] # "SFRD" or "OIII"
    ZMIN = 5.
    RSD_MODE = 1
    shot_noise = bool(int(sys.argv[6])) # 1 or 0
    quadratic_SFRD_lognormal = bool(int(sys.argv[7])) # 1 or 0
    
    ### user parameters
    UP = olim.User_Parameters(precisionboost= 1., FLAG_FORCE_LINEAR_CF= 0, MIN_R_NONLINEAR= 0.5, MAX_R_NONLINEAR= 200.0, FLAG_DO_DENS_NL= False, FLAG_WF_ITERATIVE= True)
    
    ### cosmological parameters
    CP, ClassyC, zeus_corr, HMFcl = olim.cosmo_wrapper(UP, olim.Cosmo_Parameters_Input(**olima.CosmoParams_input_fid))
    
    ### astro parameters fiducial
    AstroParams_input_fid_use = dict(astromodel = 0, accretion_model = 0, alphastar = 0.5, betastar = -0.5, epsstar = 0.1, 
                                     Mc = 3e11, Mturn_fixed = None, dlog10epsstardz = 0.0, quadratic_SFRD_lognormal = quadratic_SFRD_lognormal, 
                                     fesc10 = 0.1, alphaesc = 0., L40_xray = 1e41/1e40, E0_xray = 500., alpha_xray = -1.0, Emax_xray_norm= 2000,
                                     Nalpha_lyA_II = 9690, Nalpha_lyA_III = 17900, FLAG_MTURN_SHARP= False,
                                     C0dust = 4.43, C1dust = 1.99, sigmaUV=0.5,
                                     USE_POPIII = False, USE_LW_FEEDBACK=False)
    fesc = float(sys.argv[8]) # float
    if fesc==-1:
        fesc=None
    LX = float(sys.argv[9]) # float
    if LX==-1:
        LX=None
    
    ### line/SFRD parameters
    if which_field=="SFRD":
        LINE_MODEL = "powerlaw"
        line_dict = {"alpha_SFR":1.} 
    elif which_field=="OIII":
        LINE_MODEL = "Yang24"
        line_dict = None
    LP_input = olim.LineParams_Input(
                LINE = "OIII", # which line
                LINE_MODEL = LINE_MODEL, # model of the line luminosity
                OBSERVABLE_LIM = "Inu", # observe intensity in Jy/sr or mK
                _R =  1., # resolution for smoothing
                sigma_LMh = 0., # stochasticity in the L-SFR relation
                shot_noise = shot_noise, # add shot noise to the power spectrum
                quadratic_lognormal = True, # use 1st or 2nd order in the SFRD and line lognormal approximation MOVE TO USER PARAMS
                line_dict = line_dict
            )
    LP = olim.Line_Parameters(LP_input,UP)
    
    ### box parameters
    Lbox = int(sys.argv[3]) # int
    Nbox = int(sys.argv[4]) # int
    seed = int(sys.argv[5]) # int
    # zvals = np.linspace(5,15,20)
    
    ### saving file names
    if fesc is None and LX is None:
        foldername = f"./data/" 
    elif fesc!=-1 and LX is None: 
        foldername = f"./data_astro/fesc{fesc}/" 
    elif fesc is None and LX!=-1:
        foldername = f"./data_astro/LX{LX}/" 
    filename_base = f"olimpus{seed}_L{Lbox}_N{Nbox}"
    foldername_sims, foldername_corr = "sims/", f"corr_nomean_{which_field}/"
    file_BMF =  f"{foldername}{foldername_sims}{filename_base}_BMF.npz"
    file_SFRD  = f"{foldername}{foldername_sims}{filename_base}_{which_field}.npz"
    file_xH = f"{foldername}{foldername_sims}{filename_base}_xH.npz"
    file_T21 = f"{foldername}{foldername_sims}{filename_base}_T21.npz"
    file_corr = f"{foldername}{foldername_corr}{filename_base}_correlations.npz"
    if not shot_noise:
        file_SFRD, file_xH, file_T21 = f"{file_SFRD[:-4]}_noshotnoise.npz", f"{file_xH[:-4]}_noshotnoise.npz", f"{file_T21[:-4]}_noshotnoise.npz"
        file_corr = f"{file_corr[:-4]}_noshotnoise.npz"
    if not quadratic_SFRD_lognormal:
        file_SFRD, file_xH, file_T21 = f"{file_SFRD[:-4]}_noquadSFRD.npz", f"{file_xH[:-4]}_noquadSFRD.npz", f"{file_T21[:-4]}_noquadSFRD.npz"
        file_corr = f"{file_corr[:-4]}_noquadSFRD.npz"

    olim.z21_utilities.print_timer(start_time,text_before="> Initialisation...done in ")
    print(f">>>>> Parameters: Overwrite files:{'True'*(OVERWRITE_FILES)+'False'*(not OVERWRITE_FILES)} - {which_field} field - Lbox={Lbox} cMpc - Nbox={Nbox} - seed={seed} - With{'out'*(not shot_noise)} shot noise - With{'out'*(not quadratic_SFRD_lognormal)} quadratic SFRD - fesc={fesc} - LX={LX} <<<<<")


    #################################
    ##### Get boxes #################
    #################################
    print("\n***** Get boxes *****")
    start_time = time.time()
    ### run oLIMpus
    AP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_corr, zeus_pk = run_LIM(UP, CP, ClassyC, AstroParams_input_fid_use, LP, HMFcl, ZMIN, RSD_MODE,
                                                                              fesc=fesc, LX=LX)
    olim.z21_utilities.print_timer(start_time,text_before="> oLIMpus...done in ")
    start_time = time.time()
    
    ### run the BMF module
    zvals = zeus_coeff.zintegral #np.linspace(5,15,20)

    BMF_class = olim.BMF(zeus_coeff, HMFcl, CP, AP, ClassyC)    
    np.savez(file_BMF,zvals=zvals, BMF=BMF_class.BMF, rvals=BMF_class.Rs_BMF, ion_frac=BMF_class.ion_frac)
    olim.z21_utilities.print_timer(start_time,text_before="> BMF module...done in ")
    start_time = time.time()
    
    ### run the reionization module
    if not os.path.isfile(file_xH) or (os.path.isfile(file_xH) and OVERWRITE_FILES):
        box_reio = reio(CP, ClassyC, zeus_corr, zeus_coeff, BMF_class, zvals, 
                        input_boxlength=Lbox, ncells=Nbox, seed=seed,
                        PRINT_TIMER=False, COMPUTE_DENSITY_AT_ALLZ=True, 
                        COMPUTE_MASSWEIGHTED=False, COMPUTE_PARTIAL_IONIZATIONS=True, COMPUTE_PARTIAL_AND_MASSWEIGHTED=False)
        xHI_box, xHIIpartVW = 1.-box_reio.ion_field_partial_allz, box_reio.ion_frac_partial
        np.savez(file_xH, zvals=zvals, xHI_box=xHI_box, xHIIpartVW=xHIIpartVW, xHIIVW=box_reio.ion_frac) 
                 #xHIIpartMW=box_reio.ion_frac_partial_massweighted,  xHIIMW=box_reio.ion_frac_massweighted)
        delete_class_attributes(BMF_class) ; del BMF_class
        delete_class_attributes(box_reio) ; del box_reio
        gc.collect()
    else:
        xHI_box = np.load(file_xH)["xHI_box"]
        xHIIpartVW = np.load(file_xH)["xHIIpartVW"]
    olim.z21_utilities.print_timer(start_time,text_before="> Reionization module...done in ")
    start_time = time.time()
    
    ### run LIM simulations
    if (not os.path.isfile(file_SFRD)) or (not os.path.isfile(file_T21)) or (os.path.isfile(file_SFRD) and os.path.isfile(file_T21) and OVERWRITE_FILES):
        SFRD_box, T21_box, T21_avg = get_boxes(LP, LIM_coeff, LIM_corr, LIM_pk, zeus_coeff, zeus_pk, zvals, 
                                               1-xHI_box, xHIIpartVW, Lbox, Nbox, seed, RSD_MODE)
        np.savez(file_SFRD, zvals=zvals, SFRD_box=SFRD_box)
        np.savez(file_T21, zvals=zvals, T21_box=T21_box, T21_avg=T21_avg)
    else:
        SFRD_box = np.load(file_SFRD)["SFRD_box"]
        T21_box = np.load(file_T21)["T21_box"]
    olim.z21_utilities.print_timer(start_time,text_before="> LIM simulations...done in ")

    #################################
    ##### Get correlations ##########
    #################################
    print("\n***** Get correlations *****")
    start_time = time.time()

    if not os.path.isfile(file_corr) or (os.path.isfile(file_corr) and OVERWRITE_FILES):
        ### compute correlations
        k_bins = np.logspace(np.log10(2*np.pi/Lbox),np.log10(2*np.pi/Lbox*Nbox),20) # k array 
        get_corr_output = get_corr(SFRD_box, 1-xHI_box, T21_box, Lbox, k_bins, zvals, WHICH_COMPUTE_XI="mcfit", Rmin_xi=0.5, Rmax_xi=np.sqrt(3)*Lbox)
        k_bins_pb, r_bins, pk_auto, pk_cross, r, pearson, xi_auto, xi_cross, pkvar_auto, pkvar_cross = get_corr_output    
        
        ### save correlations
        np.savez(file_corr, k_bins_pb=k_bins_pb, r_bins=r_bins, 
                 pk_SFRD=pk_auto[0], pk_xHII=pk_auto[1], pk_xHI=pk_auto[2], pk_T21=pk_auto[3],
                 pk_SFRDxHII=pk_cross[0], pk_SFRDxHI=pk_cross[1], pk_SFRDT21=pk_cross[2],
                 r_SFRDxHII=r[0], r_SFRDxHI=r[1], r_SFRDT21=r[2],
                 pearson_SFRDxHII=pearson[0], pearson_SFRDxHI=pearson[1], pearson_SFRDT21=pearson[2],
                 xi_SFRD=xi_auto[0], xi_xHII=xi_auto[1], xi_xHI=xi_auto[2], xi_T21=xi_auto[3],
                 xi_SFRDxHII=xi_cross[0], xi_SFRDxHI=xi_cross[1], xi_SFRDT21=xi_cross[2],
                 pkvar_SFRD=pkvar_auto[0], pkvar_xHII=pkvar_auto[1], pkvar_xHI=pkvar_auto[2], pkvar_T21=pkvar_auto[3],
                 pkvar_SFRDxHII=pkvar_cross[0], pkvar_SFRDxHI=pkvar_cross[1], pkvar_SFRDT21=pkvar_cross[2])
        olim.z21_utilities.print_timer(start_time,text_before="> Correlations...done in ")

    
    olim.z21_utilities.print_timer(global_start_time,text_before="***** Done in ",text_after=" *****")













