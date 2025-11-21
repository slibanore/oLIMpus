"""
Make LIM maps! 
Author: Sarah Libanore
BGU - April 2025
"""

import numpy as np 
import powerbox as pbox
from scipy.interpolate import interp1d
from oLIMpus import z21_utilities, sfrd, LineLuminosity, cosmology
try:
    from oLIMpus import BMF
except:
    print('The reionization.py file is not yet public')
from oLIMpus.zeus21_local.zeus21.maps import reionization_maps as reio
import numexpr as ne 
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from tqdm import trange, tqdm
import pickle
import os
from matplotlib import colors as cc 
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt 

min_value = -50
max_value = 40
mid_point = abs(min_value)/(abs(min_value)+abs(max_value))
colors_list = [(0, 'cyan'),
            (mid_point/1.5, 'blue'),
            (mid_point, "black"),
            ((1.+mid_point)/2.2, 'red'),
            (1, 'yellow')]
eor_colour = cc.LinearSegmentedColormap.from_list("eor_colour",colors_list)

summer_cmap = cm.summer
colors_list = [(0, "black"),
    (0.1, summer_cmap(0)),    
    (0.2, summer_cmap(50)),
    (0.5, summer_cmap(150)),
    (1, summer_cmap(255))]     
LIM_colour_1 = cc.LinearSegmentedColormap.from_list("LIM_colour_1",colors_list)

winter_cmap = cm.winter
colors_list = [(0, "black"),
    (0.1, winter_cmap(0)),    
    (0.2, winter_cmap(50)),
    (0.5, winter_cmap(150)),
    (1, winter_cmap(255))]     
LIM_colour_2 = cc.LinearSegmentedColormap.from_list("LIM_colour_1",colors_list)

class CoevalBox_LIM_analytical:
    "Class that calculates and keeps coeval maps, one z at a time."
    "The computation is done analytically based on the estimated density and LIM power spectra"

    def __init__(self, LIM_coeff, LIM_corr, LIM_Power_Spectrum, Line_Parameters, z, input_Resolution, Lbox=600, Nbox=200, seed=1605, RSD=0, get_density_box = True):

        zlist = LIM_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.z = zlist[_iz]
        
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed

        self.Inu_bar = LIM_coeff.Inu_bar[_iz]
        klist = LIM_Power_Spectrum.klist_PS

        Resolution = max(input_Resolution, Line_Parameters._R, Lbox/Nbox)
        #if Resolution != input_Resolution:
        #    print('The resolution cannot be smaller than R and Lbox/Nbox')
        #    print('Smoothing R changed to ' + str(Resolution))

        # Produce density box
        if get_density_box:
            Pm = np.outer(LIM_Power_Spectrum.lin_growth**2, LIM_corr._PklinCF) [_iz,:]

            Pm_interp = interp1d(klist,Pm,fill_value=0.0,bounds_error=False)

            pb_delta = pbox.PowerBox(
                N=self.Nbox,                     
                dim= 3,                     
                pk = lambda k: Pm_interp(k), 
                boxlength = self.Lbox,           
                seed = self.seed,
            )

        if RSD == 0:
            Pnu = LIM_Power_Spectrum._Pk_LIM[_iz,:]
        else:
            Pnu = LIM_Power_Spectrum._Pk_LIM_RSD[_iz,:]

        Pnu_interp = interp1d(klist,Pnu,fill_value=0.0,bounds_error=False)
        if Line_Parameters.LINE_MODEL == 'powerlaw':

            pb = pbox.LogNormalPowerBox(
                N=self.Nbox,                     
                dim= 3,                     
                pk = lambda k: Pnu_interp(k)*1e10, 
                boxlength = self.Lbox,           
                seed = self.seed,
            )
            self.Inu_box_noiseless = pb.delta_x()/1e5 + self.Inu_bar

        else:
            # produce Gaussian box with boosted power spectrum
            pb = pbox.LogNormalPowerBox(
                N=self.Nbox,                     
                dim= 3,                     
                pk = lambda k: Pnu_interp(k), 
                boxlength = self.Lbox,           
                seed = self.seed,
            )
            self.Inu_box_noiseless = pb.delta_x() + self.Inu_bar


        # create shot noise box
        if Line_Parameters.shot_noise:

            Pshot_interp = lambda k: LIM_coeff.shot_noise[_iz]

            pb_shot = pbox.PowerBox(
                N=self.Nbox,                     
                dim= 3,                     
                pk = lambda k: Pshot_interp(k), 
                boxlength = self.Lbox,           
                seed = self.seed+2, # uncorrelated from the density field
            )

            self.shotnoise_box = pb_shot.delta_x() + LIM_coeff.shot_noise[_iz] # shot noise box
        else:
            self.shotnoise_box = np.zeros_like(self.Inu_box_noiseless)

        # LIM box with shot noise
        self.Inu_box = self.Inu_box_noiseless + self.shotnoise_box

        # smooth the box over R 
        klistfftx = np.fft.fftfreq(self.Inu_box.shape[0],Lbox/Nbox)*2*np.pi
        klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        Inu_noiseless_fft = np.fft.fftn(self.Inu_box_noiseless)
        Inu_fft = np.fft.fftn(self.Inu_box)

        self.Inu_box_noiseless_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, Inu_noiseless_fft))
        self.Inu_box_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, Inu_fft))

        if get_density_box:

            self.density_box = pb_delta.delta_x() # density box
            density_fft = np.fft.fftn(self.density_box)
            self.density_box_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, density_fft))


class CoevalBox_percell:
    "Produce maps by computing the LIM signal cell by cell"

    def __init__(self, LIM_coeff, LIM_corr, LIM_Power_Spectrum, Zeus_coeff, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, z, input_Resolution, Lbox=600, Nbox=200, seed=1605):

        zlist = LIM_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.z = zlist[_iz]
        
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed

        self.Inu_bar = LIM_coeff.Inu_bar[_iz]

        Resolution = max(input_Resolution, Line_Parameters._R, Lbox/Nbox)
        #if Resolution != input_Resolution:
        #    print('The resolution cannot be smaller than R and Lbox/Nbox')
        #    print('Smoothing R changed to ' + str(Resolution))

        # get density box 
        density_box_3d = CoevalBox_LIM_analytical(LIM_coeff, LIM_corr, LIM_Power_Spectrum, Line_Parameters, z, input_Resolution, Lbox, Nbox, seed, RSD=0,get_density_box=True).density_box
        density_box = density_box_3d.flatten()

        # compute the local dndM through EPS and HMF
        deltaArray = ne.evaluate('density_box')

        delta_crit_ST = Cosmo_Parameters.delta_crit_ST
        a_corr_EPS = Cosmo_Parameters.a_corr_EPS

        variance = np.var(density_box)
        sigmaR = ne.evaluate('sqrt(variance)')

        mArray, deltaArray_Mh = np.meshgrid(HMF_interpolator.Mhtab, deltaArray, indexing = 'ij', sparse = True)

        sigmaM = HMF_interpolator.sigmaintlog((np.log(mArray),self.z))

        modSigmaSq = ne.evaluate('sigmaM*sigmaM - sigmaR*sigmaR')
        indexTooBig = (modSigmaSq <= 0.0)
        modSigmaSq[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigma = ne.evaluate('sqrt(modSigmaSq)')

        nu0 = ne.evaluate('delta_crit_ST / sigmaM')
        nu0[indexTooBig] = 1.0
        modd = ne.evaluate('delta_crit_ST - deltaArray_Mh')
        nu = ne.evaluate('modd / modSigma')

        EPS_HMF_corr = ne.evaluate('(nu/nu0) * (sigmaM/modSigma)* (sigmaM/modSigma) * exp(-a_corr_EPS * (nu*nu-nu0*nu0)/2. )')
        #print('Done EPS corr in ' + str(time.time() - start))

        HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(mArray),self.z)))

        # ---- #
        # produce SFRD box
        SFRtab_currII = sfrd.SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, self.z, self.z)

        integrand = EPS_HMF_corr *  HMF_curr * SFRtab_currII * HMF_interpolator.Mhtab[:,np.newaxis]

        SFRDbox_flattend = np.trapezoid(integrand, HMF_interpolator.logtabMh, axis = 0)

        SFRDbox_Lagrangian_flattened = ne.evaluate('SFRDbox_flattend')

        SFRDbox_flattend_scaled = ne.evaluate('SFRDbox_Lagrangian_flattened * (1+density_box)')

        self.SFRD_box = SFRDbox_flattend_scaled.reshape(Nbox,Nbox,Nbox)

        # ---- #
        # LIM box
        integrand_LIM = EPS_HMF_corr * HMF_curr * LineLuminosity(SFRtab_currII, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, self.z)  * HMF_interpolator.Mhtab[:,np.newaxis]

        rhoLbox_flattened = np.trapezoid(integrand_LIM, HMF_interpolator.logtabMh, axis = 0) 

        rhoLbox_Lagrangian_flattened = ne.evaluate('rhoLbox_flattened')
         
        rhoLbox_flattend_scaled = ne.evaluate('rhoLbox_Lagrangian_flattened * (1+density_box)')

        self.rhoL_box = rhoLbox_flattend_scaled.reshape(Nbox,Nbox,Nbox)

        # get observed box 
        self.Inu_box_noiseless = self.rhoL_box * LIM_coeff.coeff1_LIM[_iz] 

        # create shot noise box -- SAME AS ANALYTICAL !!! 
        if Line_Parameters.shot_noise:

            Pshot_interp = lambda k: LIM_coeff.shot_noise[_iz]

            pb_shot = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: Pshot_interp(k), 
                boxlength = self.Lbox,           
                seed = self.seed+3, # uncorrelated from the density field
            )

            self.shotnoise_box = pb_shot.delta_x() # shot noise box
        else:
            self.shotnoise_box = np.zeros_like(self.Inu_box_noiseless)

        # LIM box with shot noise
        self.Inu_box = self.Inu_box_noiseless + self.shotnoise_box

        # smooth the box over R 
        klistfftx = np.fft.fftfreq(self.Inu_box.shape[0],Lbox/Nbox)*2*np.pi
        klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        rhoL_fft = np.fft.fftn(self.rhoL_box)
        Inu_noiseless_fft = np.fft.fftn(self.Inu_box_noiseless)
        Inu_fft = np.fft.fftn(self.Inu_box)

        self.rhoL_box_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, rhoL_fft))
        self.Inu_box_noiseless_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, Inu_noiseless_fft))
        self.Inu_box_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, Inu_fft))

        self.density_box = density_box_3d
        density_fft = np.fft.fftn(self.density_box)
        self.density_box_smooth = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, density_fft))


def get_reio_field(input_z, zeus_coeff, zeus_corr, Astro_Parameters, Cosmo_Parameters, ClassyCosmo, HMF_interpolator, Lbox=600, Nbox=200, Rmin_bubbles=0.05, seed=1605, compute_mass_weighted_xHII=False,compute_include_partlion=True,compute_partial_and_massweighted=True):

    BMF_val = BMF(zeus_coeff, HMF_interpolator, Cosmo_Parameters, Astro_Parameters, ClassyCosmo, R_linear_sigma_fit_input=10, FLAG_converge=True, max_iter=10, ZMAX_REION = 30,Rmin=Rmin_bubbles)

    box_reio = reio(Cosmo_Parameters, ClassyCosmo, zeus_corr, zeus_coeff, BMF_val, input_z, 
                 input_boxlength=Lbox, ncells=Nbox, seed=seed, r_precision=1., Rs=None, barrier=None, 
                 PRINT_TIMER=False, 
                 LOGNORMAL_DENSITY=False, COMPUTE_DENSITY_AT_ALLZ=True, SPHERIZE=False, 
                 COMPUTE_MASSWEIGHTED=compute_mass_weighted_xHII, lowres_massweighting=1, COMPUTE_PARTIAL_IONIZATIONS=compute_include_partlion,
                 COMPUTE_PARTIAL_AND_MASSWEIGHTED=compute_partial_and_massweighted, COMPUTE_ZREION=False
                )
    
    # if compute_include_partlion and compute_mass_weighted_xHII:
    #     compute_partial_and_massweighted = True

    if compute_partial_and_massweighted:
        reionization_map = box_reio.ion_field_partial_massweighted_allz
        ion_frac = box_reio.ion_frac_partial_massweighted

    else:
        if compute_include_partlion:
            reionization_map = box_reio.ion_field_partial_allz
            ion_frac = box_reio.ion_frac_partial
        else:
            reionization_map = box_reio.ion_field_allz
            if compute_mass_weighted_xHII:
                ion_frac = box_reio.ion_frac_massweighted
            else:
                ion_frac = box_reio.ion_frac

    return reionization_map, ion_frac


class CoevalMaps_T21noreionization:
    "Class that calculates and keeps coeval maps, one z at a time."

    def __init__(self, T21_coefficients, Power_Spectrum, z, Lbox=600, Nbox=200, seed=1605):
        'the KIND flag determines the kind of map you make. Options are:'
        'KIND = 0, only T21 lognormal. OK approximation'
        'KIND = 1, density and T21 correlated. T21 has a gaussian and a lognormal component. Decent approximation'
        'KIND = 2, all maps'
        'KIND = 3, same as 2 but integrating over all R. Slow but most accurate'

        zlist = T21_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z

        self.T21global_noR = T21_coefficients.T21avg_noR[_iz] 
        

        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        klist = Power_Spectrum.klist_PS
        k3over2pi2 = klist**3/(2*np.pi**2)

        Dsq_T21_lin = Power_Spectrum.Deltasq_T21_lin[_iz] 

        Dsq_dT21 = Power_Spectrum.Deltasq_dT21[_iz]

        Dsq_T21 = Power_Spectrum.Deltasq_T21[_iz]

        Pd = Power_Spectrum.Deltasq_d_lin[_iz,:]/k3over2pi2
        #Pdinterp = interp1d(klist,Pd,fill_value=0.0,bounds_error=False)
        Pd_spl = spline(np.log(klist), np.log(Pd))

        pb = pbox.PowerBox(
            N=self.Nbox,                     
            dim=3,                     
            #pk = lambda k: Pdinterp(k), 
            pk = lambda k: np.exp(Pd_spl(np.log(k))), 
            boxlength = self.Lbox,           
            seed = self.seed               
        )

        self.deltamap = pb.delta_x() #density map, basis of this KIND of approach

        #then we make a map of the linear T21 fluctuation, better to use the cross to keep sign, at linear level same 
        PdT21 = Dsq_dT21/k3over2pi2

        #powerratioint = interp1d(klist,PdT21/Pd,fill_value=0.0,bounds_error=False)
        powerratio_spl = spline(klist, PdT21/Pd) #cross can be negative, so can't interpolate over log values


        deltak = pb.delta_k()

        powerratio = powerratio_spl(pb.k())
        T21lin_k = powerratio * deltak
        self.T21maplin= self.T21global_noR + z21_utilities.powerboxCtoR(pb,mapkin = T21lin_k)

        #now make a nonlinear correction, built as \sum_R [e^(gR dR) - gR dR]. Uncorrelatd with all dR so just a separate field!
        #NOTE: its not guaranteed to work, excess power can be negative in some cases! Not for each component xa, Tk, but yes for T21
        excesspower21 = (Dsq_T21-Dsq_T21_lin)/k3over2pi2

        lognormpower = interp1d(klist,excesspower21/self.T21global_noR**2,fill_value=0.0,bounds_error=False)
        #G or logG? TODO revisit
        pbe = pbox.LogNormalPowerBox(
            N=self.Nbox,                     
            dim=3,                     
            pk = lambda k: lognormpower(k), 
            boxlength = self.Lbox,           
            seed = self.seed+1                # uncorrelated
        )

        self.T21mapNL = self.T21global_noR*pbe.delta_x()

        #and finally, just add them together!
        self.T21map = self.T21maplin +  self.T21mapNL



class CoevalBox_T21reionization:
    "Re-build the 21cm temperature map combining the xalpha, Tk and delta for more stability in the non-linear fluctuation computation. Include the xH contribution"

    def __init__(self, zeus_coeff, zeus_pk, z, reionization_map_partial, ion_frac_withpartial, Lbox=600, Nbox=200, seed=1605, MAP_T21_FULL = True,  input_Resolution = None):
        
        self.ion_frac = ion_frac_withpartial
        self.xH_avg_map = 1. - self.ion_frac
        
        self.xH_box = 1. - reionization_map_partial

        if MAP_T21_FULL:

            zeus_box = CoevalMaps_T21noreionization(zeus_coeff, zeus_pk, z, Lbox, Nbox, seed=seed)

            self.T21_map_only = zeus_box.T21map 
 
        else:
            print('T21 by ingredients not yet implemented') 
            # self.xa_map = 
            # self.invTcol_map = 

            #self.T21_map_only = cosmology.T021(Cosmo_Parameters,zeus_coeff.zintegral) * self.xa_map/(1.0 + self.xa_map) * (1.0 - zeus_coeff.T_CMB * (self.invTcol_map)) 

        self.T21_map = self.T21_map_only * self.xH_box

        if input_Resolution != None:
            Resolution = max(input_Resolution, Lbox/Nbox)
        else:
            Resolution = Lbox/Nbox

        klistfftx = np.fft.fftfreq(self.T21_map.shape[0],Lbox/Nbox)*2*np.pi
        klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        T21_fft = np.fft.fftn(self.T21_map)
        self.T21_map = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, T21_fft))
        T21_map_only_fft = np.fft.fftn(self.T21_map_only)
        self.T21_map_only = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, T21_map_only_fft))
        xH_box_fft = np.fft.fftn(self.xH_box)
        self.xH_box = np.array(z21_utilities.tophat_smooth(Resolution, klist3Dfft, xH_box_fft))


def build_lightcone(which_lightcone,
             input_zvals,
             Lbox, 
             Ncell, 
             R,
             seed, 
             analytical, 
            correlations_21,
            coefficients_21,
            PS21,
            correlations,
            coefficients,
            PSLIM,
            Rmin_bubbles,
            compute_mass_weighted_xHII,
            compute_include_partlion,
            compute_partial_and_massweighted,
            LineParams1,
            AstroParams, 
            CosmoParams,
            HMFintclass,
            ClassyCosmo,
            folder=None,
             include_label = '', 
            RSD=0
             ):

    if folder is None:
        save_path = os.path.join(os.getcwd(), "oLIMpus")
        folder_out = os.path.abspath(os.path.join(save_path, "..", 'analysis_' + str(Lbox) + ',' + str(Ncell) + ',' + str(R) ))

        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        folder = folder_out + '/lightcones'
        if not os.path.exists(folder):
            os.makedirs(folder)

    filename_all = folder + which_lightcone + include_label + '.pkl'
    print(filename_all)
    if os.path.exists(filename_all):
        with open(filename_all, 'rb') as handle:
            lightcone = pickle.load(handle)
            return lightcone
    print('Running lightcone...')
    zvals = input_zvals[::-1]
    z_long = np.linspace(zvals[0],zvals[-1],1000)
    lightcone = np.zeros((Ncell, Ncell, len(z_long)))

    box = []
    reionization_map_partial, ion_frac_withpartial = get_reio_field(
    zvals, coefficients_21, correlations_21, AstroParams, CosmoParams, ClassyCosmo, HMFintclass, Lbox, Ncell, Rmin_bubbles, seed, compute_mass_weighted_xHII,compute_include_partlion,compute_partial_and_massweighted)
    for zi in tqdm(zvals):

        box.append(lightcone_single_z(zi, zvals, Lbox,Ncell,R,seed,which_lightcone, analytical, coefficients,correlations, PSLIM, coefficients_21, PS21, reionization_map_partial[list(zvals).index(zi)], ion_frac_withpartial[list(zvals).index(zi)], HMFintclass,CosmoParams,AstroParams,LineParams1,RSD))
        
    lightcone[:, :, 0] = box[0][:, :, 0]        
    # Loop over each z in z_long
    for z_idx, zi in (enumerate(z_long[1:],start=1)):
        # Find which two matrices to interpolate between
        idx = np.searchsorted(zvals, zi) - 1
        idx = np.clip(idx, 0, len(zvals) - 2)  # Keep index within bounds
        
        z1, z2 = zvals[idx], zvals[idx + 1]
        mat1, mat2 = box[idx], box[idx + 1]
        
        # Interpolation weight
        w = (zi - z1) / (z2 - z1)
        
        # Interpolate between contiguous slices
        lightcone[:, :, z_idx] = (1 - w) * mat1[:, :, z_idx % Ncell] + w * mat2[:, :, z_idx % Ncell]

    lightcone[np.isnan(lightcone)] = 0.

    with open(filename_all, 'wb') as handle:
        pickle.dump(lightcone,handle)
    

    return lightcone


def lightcone_single_z(zi, zvals, Lbox, Nbox, Resolution, seed, which_lightcone, analytical, LIM_coeff, LIM_corr, PSLIM, coefficients_21, PS21, reionization_map_partial, ion_frac_withpartial, HMFintclass, CosmoParams,AstroParams,LineParams,RSD=0):

    if which_lightcone == 'T21' or which_lightcone == 'T21_only':
        if analytical and zi == zvals[0]:
            print('Warning! The bubble part is not analytical')
        else:
            if zi == zvals[0]:
                print('Warning! The T21 map is only  analytical, except for the bubble part')

        if which_lightcone == 'T21_only':
            box = CoevalBox_T21reionization(coefficients_21,PS21,zi,reionization_map_partial, ion_frac_withpartial,Lbox,Nbox,seed,MAP_T21_FULL=True,).T21_map_only
        else:
            box = CoevalBox_T21reionization(coefficients_21,PS21,zi,reionization_map_partial, ion_frac_withpartial,Lbox,Nbox,seed,MAP_T21_FULL=True,).T21_map

    elif which_lightcone == 'density':
        if not analytical and zi == zvals[0]:
            print('Warning! The density map is only  analytical')

        box = CoevalBox_LIM_analytical(LIM_coeff, LIM_corr, PSLIM, LineParams, zi, Resolution, Lbox, Nbox, seed, RSD, True).density_box

    elif which_lightcone == 'xHI':
        if analytical and zi == zvals[0]:
            print('Warning! The xHI map cannot be computed analytically')

        box = CoevalBox_T21reionization(coefficients_21,PS21,zi,reionization_map_partial, ion_frac_withpartial,Lbox,Nbox,seed,MAP_T21_FULL=True).xH_box

    elif which_lightcone == 'SFRD':
        box = CoevalBox_percell( LIM_coeff, LIM_corr, PSLIM, coefficients_21, LineParams, AstroParams, CosmoParams, HMFintclass, zi, Resolution, Lbox, Nbox, seed).SFRD_box

    elif which_lightcone == 'rho_L':
        box = CoevalBox_percell( LIM_coeff, LIM_corr, PSLIM, coefficients_21, LineParams, AstroParams, CosmoParams, HMFintclass, zi, Resolution, Lbox, Nbox, seed).rhoL_box

    else:

        if analytical:
            all_boxes = CoevalBox_LIM_analytical(LIM_coeff, LIM_corr, PSLIM, LineParams, zi, Resolution, Lbox, Nbox, seed, RSD,False)

        else:
            all_boxes = CoevalBox_percell( LIM_coeff, LIM_corr, PSLIM, coefficients_21, LineParams, AstroParams, CosmoParams, HMFintclass, zi, Resolution, Lbox, Nbox, seed)

        if which_lightcone == 'LIM':
            box = all_boxes.Inu_box_noiseless
        elif which_lightcone == 'LIM_SN':
            box = all_boxes.Inu_box
        elif which_lightcone == 'LIM_smooth':
            box = all_boxes.Inu_box_smooth

        else:
            print('Check lightcone')
            return 

    return box

def plot_lightcone(which_lightcone,
             input_zvals,
             Lbox, 
             Ncell, 
             R,
             seed, 
             analytical, 
            correlations_21,
            coefficients_21,
            PS21,
            correlations,
            coefficients,
            PSLIM,
            Rmin_bubbles,
            compute_mass_weighted_xHII,
            compute_include_partlion,
            compute_partial_and_massweighted,
            LineParams,
            AstroParams, 
            CosmoParams,
            HMFintclass,
            ClassyCosmo,
            RSD,
            folder=None,       
            include_label='',
            input_text_label = None,
             _islice = 0,
            ax = None,
            fig = None,
            cmap = None,
            **kwargs
            ):

    zvals = input_zvals[::-1]

    lightcone = build_lightcone(which_lightcone,
             input_zvals,
             Lbox, 
             Ncell, 
             R,
             seed, 
             analytical, 
            correlations_21,
            coefficients_21,
            PS21,
            correlations,
            coefficients,
            PSLIM,
            Rmin_bubbles,
            compute_mass_weighted_xHII,
            compute_include_partlion,
            compute_partial_and_massweighted,
            LineParams,
            AstroParams, 
            CosmoParams,
            HMFintclass,
            ClassyCosmo,
            folder=folder,
            RSD = RSD,
             include_label = include_label, 
             )
    
    if which_lightcone == 'density':
        text_label_helper = r'$\delta$'
        use_cmap = 'magma'
        vmin = -0.6
        vmax = 0.6
    elif which_lightcone == 'SFRD':
        text_label_helper = r'$\rm SFRD\,[M_\odot\,{\rm /yr/Mpc^{3})}]$'
        use_cmap = 'bwr'
        vmin = 1e-3
        vmax = 1e0
    elif which_lightcone == 'xHI':
        text_label_helper = r'$x_{\rm HI}$'
        use_cmap = 'gray'
        vmin = 0.
        vmax = 1.
    elif which_lightcone == 'T21' or which_lightcone == 'T21_only':
        text_label_helper = r'$T_{21}\,[{\mu\rm K}]$'
        use_cmap = eor_colour
        vmin = min_value
        vmax = max_value
    elif which_lightcone == 'LIM':
        text_label_helper = r'$I_{\rm %s}\,[{\rm Jy/sr}]$'%LineParams.LINE
        use_cmap = LIM_colour_1 if LineParams.LINE == 'OIII' else LIM_colour_2
        vmin = 0.
        vmax = 0.5*np.max(lightcone)
    else:
        print('Check lightcone')
        return 

    if cmap:
        use_cmap = cmap

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))

    if input_text_label is None:
        text_label = text_label_helper
    else:
        text_label = text_label_helper + ',\,' + input_text_label

    extent = [zvals[0], zvals[-1], 0, Lbox]

    if which_lightcone == 'SFRD':
        im = ax.imshow(lightcone[:,_islice,:], aspect='auto', extent=extent, cmap=use_cmap, origin='lower', norm = LogNorm(vmin=vmin, vmax=vmax))    
    else:
        im = ax.imshow(lightcone[:,0,:], aspect='auto', extent=extent, cmap=use_cmap, origin='lower', vmin = vmin,vmax=vmax)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0e")
    ax.set_ylabel(text_label,fontsize=15)
    ax.set_xlabel(r'$z$',fontsize=15)

    plt.tight_layout()

    return 