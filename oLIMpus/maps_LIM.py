"""
Make LIM maps! 
Author: Sarah Libanore
BGU - April 2025
"""

import numpy as np 
import powerbox as pbox
from scipy.interpolate import interp1d
from oLIMpus import z21_utilities, sfrd, LineLuminosity, CoevalMaps, BMF, cosmology
from oLIMpus.zeus21_local_sarah.zeus21.maps import reionization_maps as reio
import numexpr as ne 
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from tqdm import trange


class CoevalBox_LIM_analytical:
    "Class that calculates and keeps coeval maps, one z at a time."
    "The computation is done analytically based on the estimated density and LIM power spectra"

    def __init__(self, LIM_coeff, LIM_corr, LIM_Power_Spectrum, Line_Parameters, z, input_Resolution, Lbox=600, Nbox=200, seed=1605):

        zlist = LIM_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.z = zlist[_iz]
        
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed

        self.Inu_bar = LIM_coeff.Inu_bar[_iz]
        klist = LIM_Power_Spectrum.klist_PS

        sphere_FACTOR = 0.620350491
        Resolution = max(input_Resolution, Line_Parameters._R, sphere_FACTOR * Lbox/Nbox)
        if Resolution != input_Resolution:
            print('The resolution cannot be smaller than R and Lbox/Nbox')
            print('Smoothing R changed to ' + str(Resolution))

        # Produce density box
        Pm = np.outer(LIM_Power_Spectrum.lin_growth**2, LIM_corr._PklinCF) [_iz,:]

        Pm_interp = interp1d(klist,Pm,fill_value=0.0,bounds_error=False)

        pb = pbox.PowerBox(
            N=self.Nbox,                     
            dim=3,                     
            pk = lambda k: Pm_interp(k), 
            boxlength = self.Lbox,           
            seed = self.seed,
        )

        self.density_box = pb.delta_x() # density box

        # scale the density box to the lognormal intensity
        # !! should go to the User_Params
        if Line_Parameters.quadratic_lognormal:
            lognormal_nu_box = np.exp(LIM_coeff.gamma_LIM[_iz] * self.density_box + LIM_coeff.gamma2_LIM[_iz] * self.density_box**2 )
        
        else:
            lognormal_nu_box = np.exp(LIM_coeff.gamma_LIM[_iz] * self.density_box) 

        # rescale the mean so to make it Eulerian
        self.Inu_box_noiseless = self.Inu_bar * lognormal_nu_box / np.mean(lognormal_nu_box)

        # create shot noise box
        if Line_Parameters.shot_noise:

            Pshot_interp = lambda k: LIM_coeff.shot_noise[_iz]

            pb_shot = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: Pshot_interp(k), 
                boxlength = self.Lbox,           
                seed = self.seed+1, # uncorrelated from the density field
            )

            self.shotnoise_box = pb_shot.delta_x() # shot noise box
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

        # check 
        sphere_FACTOR = 0.620350491
        Resolution = max(input_Resolution, Line_Parameters._R, sphere_FACTOR * Lbox/Nbox)
        if Resolution != input_Resolution:
            print('The resolution cannot be smaller than R and Lbox/Nbox')
            print('Smoothing R changed to ' + str(Resolution))

        # get density box 
        density_box_3d = CoevalBox_LIM_analytical(LIM_coeff, LIM_corr, LIM_Power_Spectrum, Line_Parameters, self.z, Resolution, Lbox, Nbox, seed).density_box
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

        # scale the mean 
        mean_SFRD_theory = Zeus_coeff.SFRD_avg[_iz] # this is in Lagrangian
        mean_SFRD_numerical = np.mean(SFRDbox_flattend)
        SFRDbox_Lagrangian_flattened = ne.evaluate('mean_SFRD_theory/mean_SFRD_numerical * SFRDbox_flattend')

        if Line_Parameters.Eulerian: # !!! move to user_params
            SFRDbox_flattend_scaled = ne.evaluate('SFRDbox_Lagrangian_flattened * (1+density_box)')
        else:
            SFRDbox_flattend_scaled = SFRDbox_Lagrangian_flattened

        self.SFRD_box = SFRDbox_flattend_scaled.reshape(Nbox,Nbox,Nbox)

        integrand_LIM = EPS_HMF_corr * HMF_curr * LineLuminosity(SFRtab_currII, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, self.z)  * HMF_interpolator.Mhtab[:,np.newaxis]

        # ---- #
        # LIM box
        rhoLbox_flattened = np.trapezoid(integrand_LIM, HMF_interpolator.logtabMh, axis = 0) 

        # scale the mean 
        mean_rhoL_theory = LIM_coeff.rhoL_avg[_iz] # this is in Lagrangian
        mean_rhoL_numerical = np.mean(rhoLbox_flattened)
        rhoLbox_Lagrangian_flattened = ne.evaluate('mean_rhoL_theory/mean_rhoL_numerical * rhoLbox_flattened')
         
        if Line_Parameters.Eulerian:
            rhoLbox_flattend_scaled = ne.evaluate('rhoLbox_Lagrangian_flattened * (1+density_box)')
        else:
            rhoLbox_flattend_scaled = rhoLbox_Lagrangian_flattened

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
                seed = self.seed+1, # uncorrelated from the density field
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



class CoevalBox_T21reionization:
    "Re-build the 21cm temperature map combining the xalpha, Tk and delta for more stability in the non-linear fluctuation computation. Include the xH contribution"

    def __init__(self, zeus_coeff, zeus_corr, zeus_pk, Astro_Parameters, Cosmo_Parameters, ClassyCosmo, HMF_interpolator, z, Lbox=600, Nbox=200, seed=1605, MAP_T21_FULL = True, mass_weighted_xHII=False):

        zlist = zeus_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        
        BMF_val = BMF(zeus_coeff, HMF_interpolator, Cosmo_Parameters, Astro_Parameters, R_linear_sigma_fit_input=10, FLAG_converge=True, max_iter=10, ZMAX_REION = 30)

        reionization_map_binary = reio(Cosmo_Parameters, ClassyCosmo, zeus_corr, zeus_coeff, BMF_val, zeus_coeff.zintegral, 
                 input_boxlength=Lbox, ncells=Nbox, seed=seed, r_precision=1., barrier=None, 
                 PRINT_TIMER=False, ENFORCE_BMF_SCALE=False, 
                 LOGNORMAL_DENSITY=False, COMPUTE_DENSITY_AT_ALLZ=True, SPHERIZE=False, 
                 COMPUTE_MASSWEIGHTED_IONFRAC=mass_weighted_xHII, lowres_massweighting=1)
        
        reionization_map_partial, ion_frac_withpartial = partial_ionize(reionization_map_binary.density_allz, reionization_map_binary.ion_field_allz, BMF_val, Cosmo_Parameters,0, mass_weighted_xHII)

        self.ion_frac = ion_frac_withpartial
        self.xH_avg_map = 1. - self.ion_frac[_iz]
        
        self.xH_box = 1. - reionization_map_partial[_iz]

        if MAP_T21_FULL:

            zeus_box = CoevalMaps(zeus_coeff, zeus_pk, z, Lbox, Nbox, KIND=1, seed=seed)

            self.T21_map_only = zeus_box.T21map
            self.T21_map_only /= zeus_coeff.xHI_avg[_iz]
        
 
        else:
            print('T21 by ingredients not yet implemented') 
            # self.xa_map = 
            # self.invTcol_map = 

            self.T21_map_only = cosmology.T021(Cosmo_Parameters,zeus_coeff.zintegral) * self.xa_map/(1.0 + self.xa_map) * (1.0 - zeus_coeff.T_CMB * (self.invTcol_map)) 

        self.T21_map_only *= self.xH_avg_map 
        self.T21_map = self.T21_map_only * self.xH_box


# !!! towards better model for reionization
def partial_ionize(dfield, ifield, BMF_class, CosmoParams, ir, mass_weighted_xHII):
    ifield_full = np.empty(ifield.shape)
    sample_delta = np.linspace(-5, 5, 201)
    
    for i in trange(ifield.shape[0]):
        nions = BMF_class.nion_delta_r_int(CosmoParams, sample_delta, ir).T[i]
        nrecs = BMF_class.nrec(CosmoParams, sample_delta, BMF_class.ion_frac).T[i]
        partial_ifield_spl = spline(sample_delta, nions/(1+nrecs))
        ifield_full[i] = np.clip(ifield[i] + partial_ifield_spl(dfield[i]), 0, 1)

    if mass_weighted_xHII:
        ion_frac_withpartial = np.average((1+dfield) * ifield_full, axis=(1, 2, 3))
    else:
        ion_frac_withpartial = np.average(ifield_full, axis=(1, 2, 3))

    return ifield_full, ion_frac_withpartial