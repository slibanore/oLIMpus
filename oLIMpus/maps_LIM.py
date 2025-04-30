"""
Make LIM maps! 
Author: Sarah Libanore
BGU - April 2025
"""

import numpy as np 
import powerbox as pbox
from scipy.interpolate import interp1d
from oLIMpus import z21_utilities

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

        self.Inu_box = self.Inu_box_noiseless + self.shotnoise_box

        # smooth the box over R 
        klistfftx = np.fft.fftfreq(self.Inu_box.shape[0],Lbox/Nbox)*2*np.pi
        klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        Inu_fft = np.fft.fftn(self.Inu_box)

        self.Inu_box_smooth = np.array(z21_utilities.tophat_smooth(Line_Parameters._R, klist3Dfft, Inu_fft))


class CoevalBox_SFRD_analytical:
    "Class that calculates and keeps coeval maps, one z at a time."
    "The computation is done analytically based on the estimated density and SFRD power spectra"

    def __init__(self, sfrd_coeff, sfrd_corr, sfrd_Power_Spectrum, Line_Parameters, z, Lbox=600, Nbox=200, seed=1605):

        zlist = sfrd_coeff.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.z = zlist[_iz]
        
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed

        self.SFRD_bar = sfrd_coeff.SFRD_avg[_iz]
        if Line_Parameters.Eulerian: # !!! should be in the UserParams
            gamma_LIM_Lagrangian = self.gamma_LIM-1.0
            if Line_Parameters.quadratic_lognormal: #  !!! should be in the UserParams

                gamma2_LIM_Lagrangian = self.gamma2_LIM + 1/2.

                _corrfactorEulerian_LIM = (1+(gamma_LIM_Lagrangian-2*gamma2_LIM_Lagrangian)*self.sigmaofRtab_LIM**2)/(1-2*gamma2_LIM_Lagrangian*self.sigmaofRtab_LIM**2)

            else:
                _corrfactorEulerian_LIM = 1.0 + gamma_LIM_Lagrangian*self.sigmaofRtab_LIM**2

            self.SFRD_bar *= _corrfactorEulerian_LIM

        klist = sfrd_Power_Spectrum.klist_PS

        Pm = np.outer(sfrd_Power_Spectrum.lin_growth**2, sfrd_corr._PklinCF) [_iz,:]

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
        if Line_Parameters.quadratic_lognormal: #  !!! should be in the UserParams
            lognormal_nu_box = np.exp(sfrd_coeff.gamma_II_index2D[_iz] * self.delta_box + sfrd_coeff.gamma2_II_index2D[_iz] * self.delta_box**2 )
        
        else:
            lognormal_nu_box = np.exp(sfrd_coeff.gamma_II_index2D[_iz] * self.delta_box) 

        # rescale the mean so to make it Eulerian
        self.SFRD_box = self.SFRD_bar * lognormal_nu_box / np.mean(lognormal_nu_box)
