"""
Make LIM maps! 
Author: Sarah Libanore
BGU - April 2025
"""

import numpy as np 
import powerbox as pbox
from scipy.interpolate import interp1d
from oLIMpus import z21_utilities, sfrd, LineLuminosity
import numexpr as ne 

class CoevalBox_LIM_analytical:
    "Class that calculates and keeps coeval maps, one z at a time."
    "The computation is done analytically based on the estimated density and LIM power spectra"

    def __init__(self, LIM_coeff, LIM_corr, LIM_Power_Spectrum, Line_Parameters, z, Lbox=600, Nbox=200, seed=1605):

        zlist = LIM_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.z = zlist[_iz]
        
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed

        self.Inu_bar = LIM_coeff.Inu_bar[_iz]
        klist = LIM_Power_Spectrum.klist_PS

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

        self.delta_box = pb.delta_x() # density box

        # scale the density box to the lognormal intensity
        # !! should go to the User_Params
        if Line_Parameters.quadratic_lognormal:
            lognormal_nu_box = np.exp(LIM_coeff.gamma_LIM[_iz] * self.delta_box + LIM_coeff.gamma2_LIM[_iz] * self.delta_box**2 )
        
        else:
            lognormal_nu_box = np.exp(LIM_coeff.gamma_LIM[_iz] * self.delta_box) 

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
        Inu_fft = np.fft.fftn(self.Inu_box)

        self.Inu_box_smooth = np.array(z21_utilities.tophat_smooth(Line_Parameters._R, klist3Dfft, Inu_fft))



class CoevalBox_percell:
    "Produce maps by computing the LIM signal cell by cell"

    def __init__(self, LIM_coeff, LIM_corr, LIM_Power_Spectrum, Zeus_coeff, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, z, Lbox=600, Nbox=200, seed=1605):

        zlist = LIM_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.z = zlist[_iz]
        
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed

        self.Inu_bar = LIM_coeff.Inu_bar[_iz]

        # get density box 
        density_box = CoevalBox_LIM_analytical(LIM_coeff, LIM_corr, LIM_Power_Spectrum, Line_Parameters, z, Lbox, Nbox, seed).delta_box

        # smooth the density over R
        klistfftx = np.fft.fftfreq(density_box.shape[0],Lbox/Nbox)*2*np.pi
        klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        density_fft = np.fft.fftn(density_box)

        smooth_density_fields_cell = (np.array(z21_utilities.tophat_smooth(Line_Parameters._R, klist3Dfft, density_fft))).flatten()

        # compute the local dndM through EPS and HMF
        deltaArray = ne.evaluate('smooth_density_fields_cell')

        delta_crit_ST = Cosmo_Parameters.delta_crit_ST
        a_corr_EPS = Cosmo_Parameters.a_corr_EPS

        variance = np.var(smooth_density_fields_cell)
        self.smooth_delta_box = smooth_density_fields_cell.reshape(Nbox,Nbox,Nbox)
        sigmaR = ne.evaluate('sqrt(variance)')

        mArray, deltaArray_Mh = np.meshgrid(HMF_interpolator.Mhtab, deltaArray, indexing = 'ij', sparse = True)

        sigmaM = HMF_interpolator.sigmaintlog((np.log(mArray),z))

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

        HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(mArray),z)))

        # ---- #
        # produce SFRD box
        SFRtab_currII = sfrd.SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, z, z)

        integrand = EPS_HMF_corr *  HMF_curr * SFRtab_currII * HMF_interpolator.Mhtab[:,np.newaxis]

        SFRDbox_flattend = np.trapezoid(integrand, HMF_interpolator.logtabMh, axis = 0)

        # scale the mean 
        mean_SFRD_theory = Zeus_coeff.SFRD_avg[_iz] # this is in Lagrangian
        mean_SFRD_numerical = np.mean(SFRDbox_flattend)
        SFRDbox_Lagrangian_flattened = ne.evaluate('mean_SFRD_theory/mean_SFRD_numerical * SFRDbox_flattend')

        if Line_Parameters.Eulerian: # !!! move to user_params
            SFRDbox_flattend_scaled = ne.evaluate('SFRDbox_Lagrangian_flattened * (1+smooth_density_fields_cell)')
        else:
            SFRDbox_flattend_scaled = SFRDbox_Lagrangian_flattened

        self.SFRD_box = SFRDbox_flattend_scaled.reshape(Nbox,Nbox,Nbox)

        integrand_LIM = EPS_HMF_corr * HMF_curr * LineLuminosity(SFRtab_currII, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, z)  * HMF_interpolator.Mhtab[:,np.newaxis]

        # ---- #
        # LIM box
        rhoLbox_flattened = np.trapezoid(integrand_LIM, HMF_interpolator.logtabMh, axis = 0) 

        # scale the mean 
        mean_rhoL_theory = LIM_coeff.rhoL_avg[_iz] # this is in Lagrangian
        mean_rhoL_numerical = np.mean(rhoLbox_flattened)
        rhoLbox_Lagrangian_flattened = ne.evaluate('mean_rhoL_theory/mean_rhoL_numerical * rhoLbox_flattened')
         
        if Line_Parameters.Eulerian:
            rhoLbox_flattend_scaled = ne.evaluate('rhoLbox_Lagrangian_flattened * (1+smooth_density_fields_cell)')
        else:
            rhoLbox_flattend_scaled = rhoLbox_Lagrangian_flattened

        rhoL_box = rhoLbox_flattend_scaled.reshape(Nbox,Nbox,Nbox)

        # get observed box 
        self.Inu_box_noiseless = rhoL_box * LIM_coeff.coeff1_LIM[_iz] 

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
