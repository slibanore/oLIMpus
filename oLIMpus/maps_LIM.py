"""
Make LIM maps! 
Author: Sarah Libanore
BGU - June 2026
"""

from oLIMpus.inputs_LIM import * 
from oLIMpus.coefficients_LIM import get_LIM_coefficients
from oLIMpus.correlations_LIM import Power_Spectra_LIM

from zeus21 import inputs
from zeus21.inputs import Cosmo_Parameters, Astro_Parameters
from zeus21.cosmology import HMF_interpolator
from dataclasses import dataclass, field as _field, InitVar
from typing import Optional
# ----------------------------------------------------------------------- #
# define colormaps 
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
# ----------------------------------------------------------------------- #

@dataclass()
class CoevalBox_LIM_analytical:
    "Class that calculates and keeps coeval maps, one z at a time."
    "The computation is done analytically based on the estimated density and LIM power spectra"

    # arguments to pass
    CosmoParams: InitVar[Cosmo_Parameters]
    LineParams: InitVar[Line_Parameters]
    CoeffStructure: InitVar[get_LIM_coefficients]
    PowerSpectra: InitVar[Power_Spectra_LIM]
    input_z: np.ndarray
    input_density: Optional[np.ndarray] = None

    # box params
    input_boxlength: float = _field(default=300.)
    ncells: int = _field(default=300)
    seed: int = _field(default=1234)
    input_Resolution: float = _field(default=0.5)

    # boxes
    density: np.ndarray = _field(init=False)
    smooth_box: bool = _field(default=False)
    density_smooth: np.ndarray = _field(init=False)
    Inu_box_noiseless: np.ndarray = _field(init=False)
    Inu_box_noiseless_smooth: np.ndarray = _field(init=False)
    Inu_box: np.ndarray = _field(init=False)
    Inu_box_smooth: np.ndarray = _field(init=False)
    shotnoise_box: np.ndarray = _field(init=False)

    # other attributes
    _klist: np.ndarray = _field(init=False)
    _k3over2pi2: np.ndarray = _field(init=False)
    Inu_bar: np.ndarray = _field(init=False)
    _Pnu: np.ndarray = _field(init=False)
    _Pd: np.ndarray = _field(init=False)

    def __post_init__(self, CosmoParams, LineParams, CoeffStructure, PowerSpectra):

        _iz = z21_utilities.find_nearest_idx(CoeffStructure.zintegral, self.input_z)
        self._klist = PowerSpectra.klist_PS
        self._k3over2pi2 = self._klist**3/(2*np.pi**2)

        self.Inu_bar = CoeffStructure.Inu_bar[_iz]

        self._Pd = np.outer(PowerSpectra.lin_growth**2, CosmoParams._PklinCF) 

        ### generate densities
        if self.input_density is not None:
            self.density = self.input_density
        else:
            self.density, pbs = generate_density_pb(_iz, self.input_boxlength, self.ncells, self.seed, self._klist, self._Pd)

        if PowerSpectra.RSD_MODE == 0:
            self._Pnu = PowerSpectra._Pk_LIM[_iz,:]
        else:
            self._Pnu = PowerSpectra._Pk_LIM_RSD[_iz,:]

        Pnu_interp = spline(self._klist,self._Pnu)

        norm = Pnu_interp(0.1)

        pb = pbox.LogNormalPowerBox(
            N=self.ncells,                     
            dim= 3,                     
            pk = lambda k: Pnu_interp(k)/norm, 
            boxlength = self.input_boxlength,           
            seed = self.seed,
        )
        self.Inu_box_noiseless = pb.delta_x()*np.sqrt(norm) + self.Inu_bar

        # create shot noise box
        if LineParams.shot_noise:

            Pshot_interp = lambda k: CoeffStructure.shot_noise[_iz]

            pb_shot = pbox.PowerBox(
                N=self.ncells,                     
                dim= 3,                     
                pk = lambda k: Pshot_interp(k), 
                boxlength = self.input_boxlength ,          
                seed = self.seed+2, # uncorrelated from the density field
            )

            self.shotnoise_box = pb_shot.delta_x() + CoeffStructure.shot_noise[_iz] # shot noise box
        else:
            self.shotnoise_box = np.zeros_like(self.Inu_box_noiseless)

        # LIM box with shot noise
        self.Inu_box = self.Inu_box_noiseless + self.shotnoise_box

        if self.smooth_box:
            # smooth the box over R 
            Resolution = max(self.input_Resolution, LineParams._R, self.input_boxlength/self.ncells)
            self.Inu_box_noiseless_smooth = z21_utilities.smooth_box(self.Inu_box_noiseless, Resolution, self.input_boxlength, self.ncells)
            
            self.Inu_box_smooth = z21_utilities.smooth_box(self.Inu_box, Resolution, self.input_boxlength, self.ncells)
            
            self.density_smooth = z21_utilities.smooth_box(self.density, Resolution, self.input_boxlength, self.ncells)


def generate_density_pb(iz, input_boxlength, ncells, seed, _klist,_Pd):

    density = np.zeros((ncells,ncells,ncells))

    Pd_spl = spline(np.log(_klist), np.log(_Pd[iz])) # density at min z
    pb = pbox.PowerBox(
        N=ncells,
        dim=3,
        pk = lambda k: np.exp(Pd_spl(np.log(k))),
        boxlength = input_boxlength,
        seed = seed
        )
    density = pb.delta_x()

    return density, pb


@dataclass()
class CoevalBox_percell:
    "Produce maps by computing the LIM signal cell by cell"

    # arguments to pass
    LineParams: InitVar[Line_Parameters]
    CosmoParams: InitVar[Cosmo_Parameters]
    AstroParams: InitVar[Astro_Parameters]
    CoeffStructure: InitVar[get_LIM_coefficients]
    PowerSpectra: InitVar[Power_Spectra_LIM]
    HMFinterp: InitVar[HMF_interpolator]

    input_z: np.ndarray
    input_density: Optional[np.ndarray] = None

    # box params
    input_boxlength: float = _field(default=300.)
    ncells: int = _field(default=300)
    seed: int = _field(default=1234)
    input_Resolution: float = _field(default=0.5)

    # boxes
    density: np.ndarray = _field(init=False)
    smooth_box: bool = _field(default=False)
    density_smooth: np.ndarray = _field(init=False)
    SFRD_box: np.ndarray = _field(init=False)
    Inu_box_noiseless: np.ndarray = _field(init=False)
    Inu_box_noiseless_smooth: np.ndarray = _field(init=False)
    Inu_box: np.ndarray = _field(init=False)
    Inu_box_smooth: np.ndarray = _field(init=False)
    shotnoise_box: np.ndarray = _field(init=False)

    # other attributes
    _klist: np.ndarray = _field(init=False)
    _k3over2pi2: np.ndarray = _field(init=False)
    Inu_bar: np.ndarray = _field(init=False)

    def __post_init__(self, CosmoParams, AstroParams, LineParams, HMFinterp, CoeffStructure, PowerSpectra):

        _iz = z21_utilities.find_nearest_idx(CoeffStructure.zintegral, self.input_z)[0]
        self._klist = PowerSpectra.klist_PS
        self._k3over2pi2 = self._klist**3/(2*np.pi**2)

        self.Inu_bar = CoeffStructure.Inu_bar[_iz]

        self._Pd = np.outer(PowerSpectra.lin_growth**2, CosmoParams._PklinCF) 

        ### generate densities
        if self.input_density is not None:
            self.density = self.input_density
        else:
            self.density, pbs = generate_density_pb(_iz, self.input_boxlength, self.ncells, self.seed, self._klist, self._Pd)

        density = self.density.flatten()

        # compute the local dndM through EPS and HMF
        deltaArray = ne.evaluate('density')

        delta_crit_ST = CosmoParams.delta_crit_ST
        a_corr_EPS = CosmoParams.a_corr_EPS

        variance = np.var(self.density)
        sigmaR = ne.evaluate('sqrt(variance)')

        mArray, deltaArray_Mh = np.meshgrid(HMFinterp.Mhtab, deltaArray, indexing = 'ij', sparse = True)

        sigmaM = HMFinterp.sigmaintlog((np.log(mArray),self.input_z))

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

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(mArray),self.input_z)))

        # ---- #
        # produce SFRD box
        SFRtab_currII = CoeffStructure.SFR(CosmoParams, AstroParams, HMFinterp, mArray, self.input_z, 2, False, False)    

        integrand = EPS_HMF_corr *  HMF_curr * SFRtab_currII * HMFinterp.Mhtab[:,np.newaxis]

        SFRDbox_flattend = np.trapezoid(integrand, HMFinterp.logtabMh, axis = 0)

        SFRDbox_Lagrangian_flattened = ne.evaluate('SFRDbox_flattend')

        SFRDbox_flattend_scaled = ne.evaluate('SFRDbox_Lagrangian_flattened * (1+density)')

        self.SFRD_box = SFRDbox_flattend_scaled.reshape(self.ncells,self.ncells,self.ncells)

        # ---- #
        # LIM box
        integrand_LIM = EPS_HMF_corr * HMF_curr * CoeffStructure.LineLuminosity(SFRtab_currII, CosmoParams, AstroParams, LineParams, HMFinterp, mArray, self.input_z)  * HMFinterp.Mhtab[:,np.newaxis]

        rhoLbox_flattened = np.trapezoid(integrand_LIM, HMFinterp.logtabMh, axis = 0) 

        rhoLbox_Lagrangian_flattened = ne.evaluate('rhoLbox_flattened')
         
        rhoLbox_flattend_scaled = ne.evaluate('rhoLbox_Lagrangian_flattened * (1+density)')

        self.rhoL_box = rhoLbox_flattend_scaled.reshape(self.ncells,self.ncells,self.ncells)

        # get observed box 
        self.Inu_box_noiseless = self.rhoL_box * CoeffStructure.coeff1_LIM[_iz] 

        # create shot noise box -- SAME AS ANALYTICAL !!! 
        if LineParams.shot_noise:

            Pshot_interp = lambda k: CoeffStructure.shot_noise[_iz]

            pb_shot = pbox.PowerBox(
                N=self.ncells,                     
                dim=3,                     
                pk = lambda k: Pshot_interp(k), 
                boxlength = self.input_boxlength,           
                seed = self.seed+3, # uncorrelated from the density field
            )

            self.shotnoise_box = pb_shot.delta_x() # shot noise box
        else:
            self.shotnoise_box = np.zeros_like(self.Inu_box_noiseless)

        # LIM box with shot noise
        self.Inu_box = self.Inu_box_noiseless + self.shotnoise_box

        # smooth the box over R 
        if self.smooth_box:

            Resolution = max(self.input_Resolution, LineParams._R, self.input_boxlength/self.ncells)

            self.rhoL_box_smooth = z21_utilities.smooth_box(self.rhoL_box, Resolution, self.input_boxlength, self.ncells)
            
            self.Inu_box_noiseless_smooth = z21_utilities.smooth_box(self.Inu_box_noiseless, Resolution, self.input_boxlength, self.ncells)

            self.Inu_box_smooth = z21_utilities.smooth_box(self.Inu_box, Resolution, self.input_boxlength, self.ncells)

            self.density_smooth = z21_utilities.smooth_box(self.density, Resolution, self.input_boxlength, self.ncells)

"""

TODO : THESE HAVE TO BE FIXED 

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
    zvals, coefficients_21, correlations_21, AstroParams, CosmoParams, ClassyCosmo, HMFintclass, Lbox, Ncell, Rmin_bubbles, seed, compute_mass_weighted_xHII,compute_include_partlion)
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
        use_cmap = LIM_colour_1 if LineParams.LINE[:4] == 'OIII' else LIM_colour_2
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
"""


"""
Generate lognormal coeval boxes (LIM intensity, xHI, T21) with arbitrary
non-cubic geometry (Lx,Ly,Lz / Nx,Ny,Nz).

Debugged/optimized version. Changes vs. previous draft are marked
### FIX n  (bugs)  and  ### OPT n  (optimizations); see accompanying notes.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from . import z21_utilities   # adjust import to your package layout


class generate_asym_boxes:

    def __init__(self,
                 CosmoParams, LineParams,
                 T21coeffs, T21PowerSpectra,
                 LIMcoeffs, LIMPowerSpectra,
                 z,
                 Lx, Ly, Lz,
                 Nx, Ny, Nz,
                 seed=1605,
                 COMPUTE_TAU=False,
                 ANISO_MODE='window',
                 bandlimit_factor=0.8,
                 COMPUTE_PARTIAL_IONIZATIONS=True,
                 T21LIN_FROM_ZEUS_PD=True,
                 r_precision=1.):
        """
        Lognormal random fields with arbitrary box geometry, for LIM and 21cm.

        ANISO_MODE : how to treat anisotropic cell resolution (dx!=dy!=dz):
            'cut'        — zero all input power at k > bandlimit_factor*min(kNy)
                           (previous behavior; isotropic guillotine).
            'window'     — multiply every input power spectrum by the
                           anisotropic pixel window squared,
                           W^2(k) = prod_i sinc^2(k_i d_i / 2),
                           mode by mode (direction-dependent, no hard cut).
            'window+cut' — both: windowed spectra plus a residual cut at
                           bandlimit_factor*min(kNy), protecting against the
                           aliasing that the lognormal exponentiation still
                           produces beyond the coarse-axis Nyquist.
            None         — no treatment (not recommended for anisotropic cells).
        Has no effect when the resolution is isotropic (the fiducial powerbox
        pipeline does not include a pixel window, so applying one to cubic
        boxes would break the cubic cross-check).
        """

        self.seed = seed
        self.COMPUTE_TAU = COMPUTE_TAU              ### FIX 1: flag was never stored -> AttributeError later

        # RNG streams. NOTE: density and LIM intentionally share the SAME
        # underlying white noise (same phases), reproducing the original
        # classes where both powerbox objects use seed=self.seed.
        self.rng      = np.random.default_rng(seed)       # density + LIM
        self.rng_nl   = np.random.default_rng(seed + 1)   # T21 nonlinear correction
        self.rng_shot = np.random.default_rng(seed + 2)   # LIM shot noise

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Ntot = Nx * Ny * Nz
        self.dx, self.dy, self.dz = Lx / Nx, Ly / Ny, Lz / Nz
        self.V = Lx * Ly * Lz

        # ---------------- k grids (built once, reused everywhere) ---------- #
        ### OPT 1: broadcast 1D arrays instead of np.meshgrid (saves ~3 full
        ### float64 grids of memory) and build the half (rfft) grid only once.
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=self.dy)
        kz_half = 2 * np.pi * np.fft.rfftfreq(Nz, d=self.dz)
        self.kx, self.ky, self.kz = kx, ky, kz_half
        self.k = np.sqrt(kx[:, None, None]**2
                         + ky[None, :, None]**2
                         + kz_half[None, None, :]**2)

        self.anisotropic_res = not (self.dx == self.dy == self.dz)
        self.k_cut = None
        self._W2 = None
        if self.anisotropic_res:
            kNy_x, kNy_y, kNy_z = np.pi/self.dx, np.pi/self.dy, np.pi/self.dz
            print('Box with asymmetric resolution. Nyquist scales:')
            ### FIX 2: kNy_y was printed as kNy_x (typo)
            print(f'kNy_x = {kNy_x:.2f}, kNy_y = {kNy_y:.2f}, kNy_z = {kNy_z:.2f}')

            if ANISO_MODE in ('window', 'window+cut'):
                ### OPTION 3: anisotropic pixel window. Each cell averages the
                ### field over a dx*dy*dz box, i.e. multiplies the spectrum by
                ### W(k) = sinc(kx dx/2) sinc(ky dy/2) sinc(kz dz/2)
                ### (sinc(x) = sin x / x; np.sinc uses sin(pi x)/(pi x), hence
                ### the /(2 pi)). Applying W^2 to the input power spectra makes
                ### the generated field represent the CELL-AVERAGED continuous
                ### field, with the correct direction-dependent suppression
                ### instead of an isotropic hard cut. NOTE this is applied per
                ### k-COMPONENT, so it needs kx, ky, kz — not just |k|.
                Wx = np.sinc(kx * self.dx / (2 * np.pi))
                Wy = np.sinc(ky * self.dy / (2 * np.pi))
                Wz = np.sinc(kz_half * self.dz / (2 * np.pi))
                self._W2 = (Wx[:, None, None]
                            * Wy[None, :, None]
                            * Wz[None, None, :])**2
                print('Applying the anisotropic pixel window W^2(k) to all '
                      'input power spectra (fields represent cell averages).')

            if ANISO_MODE in ('cut', 'window+cut'):
                self.k_cut = bandlimit_factor * min(kNy_x, kNy_y, kNy_z)
                print(f'Bandlimiting all input power spectra at k_cut = {self.k_cut:.2f} '
                      '(modes near the smallest Nyquist frequency set to zero).')

        # ---------------- redshift bookkeeping ---------------------------- #
        ### OPT 2: use argmin instead of min(range(...), key=lambda ...)
        zlist_LIM = LIMcoeffs.zintegral
        _iz = int(np.argmin(np.abs(zlist_LIM - z)))
        self.z = zlist_LIM[_iz]

        self.z_21 = np.atleast_1d(z)
        self._z21_idx = np.arange(len(self.z_21))
        self.z_of_density = self.z_21[0]

        self._has_density = False
        self._has_p = False
        self._has_mwp = False                      ### FIX 3: was never initialized -> AttributeError in COMPUTE_TAU path

        # shared white noise for density & LIM (Hermitian by construction)
        ### FIX 4: draw white noise in real space and rfftn it. The previous
        ### direct (a+ib) draw on the half grid was not Hermitian-symmetric on
        ### the kz=0 and kz=Nyquist planes; irfftn silently symmetrized it,
        ### halving the power of those modes. <|w_k|^2> = Ntot.
        self._w_k = np.fft.rfftn(self.rng.normal(size=(Nx, Ny, Nz)))

        # =================================================================== #
        # LIM box
        # =================================================================== #
        self.Inu_bar = LIMcoeffs.Inu_bar[_iz]
        klist_LIM = LIMPowerSpectra.klist_PS

        if LIMPowerSpectra.RSD_MODE == 0:
            Pnu = LIMPowerSpectra._Pk_LIM[_iz, :] / self.Inu_bar**2
        else:
            Pnu = LIMPowerSpectra._Pk_LIM_RSD[_iz, :] / self.Inu_bar**2

        Pnu_interp = interp1d(klist_LIM, Pnu, fill_value=0.0, bounds_error=False)

        ### FIX 5: previously the bandlimit was applied to the (unused)
        ### half-grid P_ln while the lognormal transform was recomputed on an
        ### un-bandlimited full grid -> the cut had NO effect. Now the cut is
        ### applied inside _gaussianized_power, to the array actually used.
        ### OPT 3: the P -> xi -> log(1+xi) -> P_g round trip is now done with
        ### rfft transforms on the half grid (P(|k|) is real and even, so
        ### irfftn/rfftn are exact) instead of complex fftn on the full grid:
        ### ~half the FFT cost and memory, no duplicate k-grid construction.
        P_g_LIM = self._gaussianized_power(Pnu_interp)

        g = np.fft.irfftn(self._color_modes(P_g_LIM, w_k=self._w_k),
                          s=(Nx, Ny, Nz))
        self.Inu_box_noiseless = np.exp(g - 0.5 * np.var(g)) * self.Inu_bar
        del g, P_g_LIM

        # shot noise
        if LineParams.shot_noise:
            Pshot = LIMcoeffs.shot_noise[_iz]
            ### OPT 4: a constant (white) power spectrum in k-space is exactly
            ### iid Gaussian noise in real space with variance Pshot*Ntot/V —
            ### no FFTs needed, and Hermitian symmetry is trivially satisfied.
            ### NOTE: shot noise is deliberately NOT pixel-windowed. It is the
            ### per-cell Poisson variance of the discrete emitters within each
            ### cell — already a cell-level quantity, white on the grid by
            ### construction — not a continuous field being cell-averaged.
            self.shotnoise_box = (self.rng_shot.normal(size=(Nx, Ny, Nz))
                                  * np.sqrt(Pshot * self.Ntot / self.V))
            self.shotnoise_box += Pshot   # mean offset kept to match original class
        else:
            self.shotnoise_box = np.zeros_like(self.Inu_box_noiseless)

        self.Inu_box = self.Inu_box_noiseless + self.shotnoise_box

        # =================================================================== #
        # density and xH box
        # =================================================================== #
        default_len = len(CosmoParams._Rtabsmoo)
        self.r_precision = r_precision

        # conservative choices for anisotropic cells:
        # smallest smoothing radius set by the COARSEST cell, largest by the
        # SHORTEST box side (the largest sphere that fits the box).
        self.dx21 = max(self.dx, self.dy, self.dz)
        self.boxlength21 = min(Lx, Ly, Lz)
        self.r = np.logspace(np.log10(self.dx21 * (3/4/np.pi)**(1/3)),
                             np.log10(self.boxlength21),
                             int(default_len * self.r_precision))
        self._r_idx = np.arange(int(default_len * self.r_precision))

        self._k21 = self.compute_k()

        delta_k_density, self.density = self.generate_density(CosmoParams)

        self.sig_corr = self.sigma_correction(CosmoParams)   ### FIX 6: pass scalar z internally
        self.density /= self.sig_corr   # non-ergodicity correction

        self.density_smoothed_allr = self.smooth_density()

        self.generate_density_allz(CosmoParams, self.z_21)

        self.barrier = T21coeffs.B(self.z_21, self.r)  # BMF linear barrier

        self.ion_field_allz, self.ion_frac = self.generate_xHII(CosmoParams, self.z_21)

        ### FIX 13: the fiducial T21_maps uses BINARY ionization by default
        ### (ReioMapsConfig.COMPUTE_PARTIAL_IONIZATIONS = False), while this
        ### class always applied partial ionizations. Partial ionizations
        ### change x̄H and P_xH, and since the large-scale 21cm power at these
        ### z is dominated by T21avg^2 * P_xH (+ the T21avg*x̄H mean term),
        ### this mismatch shifts the 21cm and cross spectra at large scales
        ### while leaving the LIM auto spectrum untouched. Set this flag to
        ### whatever the fiducial run used.
        self.COMPUTE_PARTIAL_IONIZATIONS = COMPUTE_PARTIAL_IONIZATIONS
        if COMPUTE_PARTIAL_IONIZATIONS or self.COMPUTE_TAU:
            ### FIX 7: z_21 was previously passed POSITIONALLY into the r slot of
            ### compute_partial(CosmoParams, CoeffStructure, r=None, z_input=None),
            ### so r became the redshift array and z_input stayed None -> crash.
            self.compute_partial(CosmoParams, T21coeffs, z_input=self.z_21)

        if COMPUTE_PARTIAL_IONIZATIONS:
            self.ion_frac = self.ion_frac_partial
            self.xH_box = 1. - self.ion_field_partial_allz[0]
        else:
            self.xH_box = (1. - self.ion_field_allz[0]).astype(np.float32)

        self.xH_avg = 1. - self.ion_frac   # scalar per z (was misleadingly called *_map)
        self.xH_box[np.isnan(self.xH_box)] = 0.

        if self.COMPUTE_TAU:
            ### FIX 8: complete rewrite, see compute_xHI_massweighted_history.
            ### The old path crashed (wrong ionize() signature, barrier only
            ### tabulated at z_21, _has_mwp undefined, wrong attribute in the
            ### return, arrays shaped for one z but indexed over all zintegral)
            ### and would have allocated (len(zintegral), Nx, Ny, Nz) boxes.
            self.xHI_massweighted = self.compute_xHI_massweighted_history(CosmoParams, T21coeffs)
            self.tau = T21coeffs.tau_reio(CosmoParams, T21coeffs.zintegral,
                                          self.xHI_massweighted)

        # =================================================================== #
        # 21cm box
        # =================================================================== #
        zlist21 = T21coeffs.zintegral
        _iz21 = int(np.argmin(np.abs(zlist21 - z)))          ### OPT 2

        klist21 = T21PowerSpectra.klist_PS
        k3over2pi2 = klist21**3 / (2 * np.pi**2)

        self.T21avg = (T21coeffs.T21avg / (T21coeffs.xHI_avg + 1e-15))[_iz21]

        Dsq_T21_lin = ((T21PowerSpectra.Deltasq_T21_lin[_iz21].T
                        / T21coeffs.T21avg[_iz21]**2) * self.T21avg**2).T
        Dsq_T21 = ((T21PowerSpectra.Deltasq_T21[_iz21].T
                    / T21coeffs.T21avg[_iz21]**2) * self.T21avg**2).T

        PdT21 = (T21PowerSpectra.Deltasq_dT21[_iz21] / T21coeffs.T21avg[_iz21]) \
                * self.T21avg / k3over2pi2
        Pd = T21PowerSpectra.Deltasq_d_lin[_iz21, :] / k3over2pi2

        # ---- linear T21 map, colored from the SAME density modes ---------- #
        ### FIX 9: the power-ratio spline was built on the LIM k grid
        ### (klist = LIMPowerSpectra.klist_PS) while PdT21 and Pd live on
        ### klist21 = T21PowerSpectra.klist_PS. Correct grid used here.
        powerratio_spl = spline(klist21, PdT21 / Pd)

        ### FIX 14: the fiducial T21_maps colors T21_lin from a density drawn
        ### with zeus21's LINEAR Pd (= Deltasq_d_lin / k3over2pi2), so its
        ### power is exactly PdT21^2/Pd. This class draws its density from
        ### ClassCosmo.pk (needed for the xH machinery); if that spectrum
        ### differs from Pd (conventions, halofit, ...), T21_lin inherits a
        ### scale-dependent offset that the LIM box does not have. With
        ### T21LIN_FROM_ZEUS_PD=True we recolor the SAME white-noise phases
        ### with Pd, reproducing the fiducial construction exactly while
        ### keeping full correlation with the density/xH fields.
        if T21LIN_FROM_ZEUS_PD:
            Pd_spl = spline(np.log(klist21), np.log(Pd))
            kk = np.where(self.k > 0, self.k, 1.0)
            P_d21 = np.exp(Pd_spl(np.log(kk)))
            P_d21 = self._apply_aniso_shaping(P_d21)
            delta_k_forT21 = self._color_modes(P_d21, w_k=self._w_k)
            del P_d21
        else:
            delta_k_forT21 = delta_k_density

        T21lin_k = powerratio_spl(self.k) * delta_k_forT21
        T21lin_k[0, 0, 0] = 0.0

        self.T21_lin = self.T21avg + np.fft.irfftn(
            T21lin_k, s=(Nx, Ny, Nz)).astype(np.float32)
        del T21lin_k

        # ---- nonlinear (lognormal) correction ----------------------------- #
        excesspower21 = (Dsq_T21 - Dsq_T21_lin) / k3over2pi2
        lognormpower = interp1d(klist21, excesspower21 / self.T21avg**2,
                                fill_value=0.0, bounds_error=False)

        ### FIX 10 (source of your power-spectrum offset): the original code
        ### uses pbox.LogNormalPowerBox, which converts the TARGET power to
        ### the power of the underlying GAUSSIAN field via
        ###     xi_target -> xi_g = ln(1 + xi_target) -> P_g,
        ### then draws Gaussian modes with P_g and exponentiates. The previous
        ### draft drew Gaussian modes directly with the TARGET power and then
        ### exponentiated, so the output field carried the target power PLUS
        ### the higher-order lognormal terms (~ xi^2/2 + ...) — a small,
        ### k-dependent excess even for cubic boxes. Same transform as the LIM
        ### box is now applied.
        P_g_21 = self._gaussianized_power(lognormpower)

        g21 = np.fft.irfftn(self._color_modes(P_g_21, rng=self.rng_nl),
                            s=(Nx, Ny, Nz))
        self.T21_NL = self.T21avg * (np.exp(g21 - 0.5 * np.var(g21)) - 1.0)
        del g21, P_g_21

        # ---- combine ------------------------------------------------------ #
        self.T21 = (self.T21_lin + self.T21_NL) * self.xH_box
        self.T21[np.isnan(self.T21)] = 0.

    # ------------------------------------------------------------------ #
    # field-generation helpers
    # ------------------------------------------------------------------ #

    def _apply_aniso_shaping(self, P_half):
        """
        Apply the anisotropic-resolution treatment to a target power spectrum
        on the rfft half grid: pixel window W^2 (mode 'window'), hard cut at
        k_cut (mode 'cut'), or both. No-op for isotropic resolution.
        """
        if self._W2 is not None:
            P_half = P_half * self._W2
        if self.k_cut is not None:
            P_half = np.where(self.k > self.k_cut, 0.0, P_half)
        return P_half

    def _gaussianized_power(self, P_interp):
        """
        Power spectrum of the Gaussian field underlying a lognormal field with
        target power P_interp:  P -> xi -> ln(1+xi) -> P_g.
        Done on the rfft half grid: P(|k|) is real and even, so the rfft round
        trip is exact.
        """
        P = np.asarray(P_interp(self.k), dtype=np.float64)
        P = self._apply_aniso_shaping(P)
        ### With the pixel window the target is no longer isotropic, but it is
        ### still real and even under k -> -k (sinc^2 is even per axis), so the
        ### rfft round trip below remains exact. Note the window is applied
        ### BEFORE Gaussianization: the generated field is the lognormal field
        ### whose measured spectrum is the windowed target. Windowing and
        ### exponentiation do not commute exactly (a truly cell-averaged
        ### lognormal field is not lognormal), so a small residual near the
        ### coarse Nyquist remains — use ANISO_MODE='window+cut' if it matters.
        np.maximum(P, 0.0, out=P)          # excess power can dip negative; clip before sqrt/log
        P[0, 0, 0] = 0.0

        xi = np.fft.irfftn(P, s=(self.Nx, self.Ny, self.Nz)) * (self.Ntot / self.V)
        if xi.min() <= -1.0:
            print('WARNING: xi <= -1 encountered in lognormal transform; clipping. '
                  'The target power spectrum may not be realizable as a lognormal field.')
            xi = np.maximum(xi, -1.0 + 1e-10)
        xi_g = np.log1p(xi)

        P_g = np.fft.rfftn(xi_g).real * (self.V / self.Ntot)
        np.maximum(P_g, 0.0, out=P_g)      # same clipping as before (Step 4)
        return P_g

    def _color_modes(self, P_half, rng=None, w_k=None):
        """
        Gaussian Fourier modes with <|delta_k|^2> = P * Ntot^2 / V, guaranteed
        Hermitian-consistent (white noise drawn in real space).
        Pass w_k to reuse an existing white-noise transform (fixed phases).
        """
        if w_k is None:
            w_k = np.fft.rfftn(rng.normal(size=(self.Nx, self.Ny, self.Nz)))
        delta_k = w_k * np.sqrt(P_half * self.Ntot / self.V)
        delta_k[0, 0, 0] = 0.0
        return delta_k

    # ------------------------------------------------------------------ #
    # density
    # ------------------------------------------------------------------ #

    def generate_density(self, CosmoParams):
        klist = CosmoParams._klistCF
        pk_matter = np.array([CosmoParams.ClassCosmo.pk(kk, self.z_of_density)
                              for kk in klist])
        pk_spl = spline(np.log(klist), np.log(pk_matter))

        kk = np.where(self.k > 0, self.k, 1.0)     # avoid log(0); zero mode nulled below
        P = np.exp(pk_spl(np.log(kk)))
        P = self._apply_aniso_shaping(P)

        ### FIX 11: previously a fresh default_rng(self.seed) was created here
        ### and new (a, b) were drawn on the half grid — non-Hermitian (FIX 4)
        ### and duplicating the k-grid construction. Reusing self._w_k keeps
        ### the density and LIM fields built from identical phases, exactly as
        ### in the original classes (both powerboxes seeded with self.seed).
        delta_k_density = self._color_modes(P, w_k=self._w_k)
        density_field = np.fft.irfftn(delta_k_density,
                                      s=(self.Nx, self.Ny, self.Nz)).astype(np.float32)
        return delta_k_density, density_field

    def generate_density_allz(self, CosmoParams, z_input):
        Dg = CosmoParams.growthint(z_input)
        growthfactor_ratio = (Dg / Dg[0])[:, None, None, None]
        self.density_allz = self.density[np.newaxis] * growthfactor_ratio
        self._has_density = True
        return self.density_allz

    def compute_k(self):
        """Full (fftn-layout) |k| grid, for real-space smoothing."""
        kz_full = 2 * np.pi * np.fft.fftfreq(self.Nz, d=self.dz)
        ### OPT 1 again: broadcasting instead of meshgrid
        return np.sqrt(self.kx[:, None, None]**2
                       + self.ky[None, :, None]**2
                       + kz_full[None, None, :]**2)

    def smooth_density(self):
        density_fft = np.fft.fftn(self.density)
        ### OPT 5: store the (len(r), Nx, Ny, Nz) stack in float32 — this is by
        ### far the largest allocation in the class.
        return np.array([z21_utilities.tophat_smooth(rr, self._k21, density_fft)
                         for rr in self.r], dtype=np.float32)

    def sigma_correction(self, CosmoParams):
        ### FIX 6: ClassCosmo.sigma expects a scalar redshift; the previous
        ### version passed the z array.
        return (np.std(self.density)
                / CosmoParams.ClassCosmo.sigma(self.r[0], self.z_of_density))

    # ------------------------------------------------------------------ #
    # ionization
    # ------------------------------------------------------------------ #

    def ionize(self, CosmoParams, curr_z_idx, z_input):
        ### FIX 12a: Dg0 must be the growth at the redshift the density was
        ### GENERATED at, not at z_input[0] (they coincide only when
        ### z_input == self.z_21; for the tau history over the full zintegral
        ### they differ).
        Dg0 = CosmoParams.growthint(self.z_of_density)
        Dg = CosmoParams.growthint(z_input[curr_z_idx])
        return np.any(self.density_smoothed_allr
                      > (Dg0 / Dg) * self.barrier[curr_z_idx, self._r_idx][:, None, None, None],
                      axis=0)

    def generate_xHII(self, CosmoParams, z_input):
        nz = len(z_input)
        ion_field_allz = np.zeros((nz, self.Nx, self.Ny, self.Nz), dtype=bool)  ### OPT 6: bool, not float64
        ion_frac = np.zeros(nz)
        for i in range(nz):
            ### FIX 12b: ionize() was called without its z_input argument -> TypeError
            ion_field_allz[i] = self.ionize(CosmoParams, self._z21_idx[i], z_input)
            ion_frac[i] = np.mean(ion_field_allz[i])
        return ion_field_allz, ion_frac

    def compute_partial(self, CosmoParams, CoeffStructure, r=None, z_input=None):
        if z_input is None:
            z_input = self.z_21
        if r is None:
            r = self.r[0]
        if not self._has_p:
            self.ion_frac_partial = np.empty(len(z_input))
            self.ion_field_partial_allz = np.empty(
                (len(z_input), self.Nx, self.Ny, self.Nz), dtype=np.float32)
        if not self._has_density:
            self.generate_density_allz(CosmoParams, z_input)

        sample_d = np.linspace(-5, 5, 51)
        out_shape = self.density.shape
        for i in range(len(z_input)):
            tempgrid = CoeffStructure.prebarrier_xHII_int_grid(sample_d, z_input[i], r)
            partialfield = np.interp(self.density.ravel(), sample_d,
                                     tempgrid).reshape(out_shape)
            np.abs(partialfield, out=partialfield)
            np.add(self.ion_field_allz[i], partialfield,
                   out=self.ion_field_partial_allz[i])
            np.clip(self.ion_field_partial_allz[i], 0, 1,
                    out=self.ion_field_partial_allz[i])

        self.ion_frac_partial = np.average(self.ion_field_partial_allz, axis=(1, 2, 3))
        self._has_p = True
        return self.ion_frac_partial, self.ion_field_partial_allz

    def compute_xHI_massweighted_history(self, CosmoParams, CoeffStructure, r=None):
        """
        Mass-weighted (partial-ionization) neutral fraction over the FULL
        CoeffStructure.zintegral grid, for tau_reio.

        ### FIX 8 / OPT 7: processes one redshift at a time and keeps only the
        scalar fraction, instead of allocating (len(zintegral), Nx, Ny, Nz)
        boxes. Mirrors the original reionization_maps recipe: binary
        ionization from the smoothed density with growth-scaled barrier,
        partial ionizations interpolated from the z0 density (as in
        compute_partial), mass weighting with the growth-evolved density.
        """
        if r is None:
            r = self.r[0]
        zgrid = CoeffStructure.zintegral
        barrier_all = CoeffStructure.B(zgrid, self.r)      # barrier on the FULL z grid
        Dg0 = CosmoParams.growthint(self.z_of_density)
        Dg = CosmoParams.growthint(zgrid)
        sample_d = np.linspace(-5, 5, 51)
        out_shape = self.density.shape

        xHI_mw = np.empty(len(zgrid))
        for i, zi in enumerate(zgrid):
            ion = np.any(self.density_smoothed_allr
                         > (Dg0 / Dg[i]) * barrier_all[i, self._r_idx][:, None, None, None],
                         axis=0)
            tempgrid = CoeffStructure.prebarrier_xHII_int_grid(sample_d, zi, r)
            partial = np.interp(self.density.ravel(), sample_d,
                                tempgrid).reshape(out_shape)
            np.abs(partial, out=partial)
            ion_partial = np.clip(ion + partial, 0., 1.)
            dens_i = self.density * (Dg[i] / Dg0)          # growth-evolved density
            xHI_mw[i] = 1.0 - np.mean((1.0 + dens_i) * ion_partial)

        self._has_mwp = True
        return xHI_mw



