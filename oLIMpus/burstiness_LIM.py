"""

Bursty star formation and its effect on the LIM shot noise.

Implements Kovetz, Lazare, Libanore, Munoz & Vanzan (2026), arXiv:2605.13967,
which propagates the M26 burstiness inference (Munoz et al. 2026,
arXiv:2601.07912) through to the LIM shot-noise power spectrum.

The model in one paragraph
-------------------------
The SFR of a galaxy in a halo of mass Mh at cosmic time t is a mean-anchored
lognormal driven by a stationary Ornstein-Uhlenbeck process,

    Mdot_*(Mh, t) = <Mdot_*>(Mh, t) exp[x(Mh, t) - sigma_x^2(Mh)/2],
    xi_x(Mh, dt)  = sigma_x^2(Mh) exp(-|dt|/tau_PS),
    sigma_x^2(Mh) = sigma_PS^2(Mh)/2,

with the M26 mass dependence sigma_PS(Mh) = sigma_PS(Mpiv) + dsigma/dlog10Mh *
(log10 Mh - log10 Mpiv). The line luminosity is the SFR convolved with the
stellar-population-synthesis Green's function of that line, approximated by a
top hat of effective width t_lambda. Because the convention anchors the MEAN,
the mean intensity, the effective bias and the clustering term are all
UNCHANGED by burstiness -- only the second moment, i.e. the shot noise, moves:

    P_shot = X_lambda^2 Int dMh (dn/dMh) Lbar_lambda^2(Mh) [1 + V_lambda(Mh)],

    V_lambda(Mh) = (2/t^2) Int_0^t ds (t-s) {exp[sigma_x^2(Mh) e^(-s/tau_PS)] - 1}

(Eqs. 5-6 of 2605.13967). For two lines observed at the same redshift the
same-halo cross moment replaces V_lambda by V_12 (Eqs. 8-10), obtained from the
overlap of the two top-hat windows.

Numerics
--------
Both V integrals are done in closed form. Expanding exp(sigma_x^2 u) as a power
series in u = exp(-|s|/tau) makes every term an elementary integral of
(window overlap) x exp(-n s / tau), so

    V_lambda = (2/t^2) Sum_n (sigma_x^2)^n / n! * I_ramp(t, n/tau)

with I_ramp(T, a) = Int_0^T (T-u) e^{-au} du = T/a - (1 - e^{-aT})/a^2. Every
term is positive, so there is no cancellation, and the coefficients are
evaluated as exp(n ln sigma_x^2 - lngamma(n+1)) so nothing overflows. This
agrees with adaptive quadrature to <= 5e-14 relative for every case in Table I
of the paper and for sigma_PS up to 5, is vectorised over halo mass, and
removes the only adaptive quadrature left in oLIMpus.

Author: Sarah Libanore
BGU - August 2026
"""

from oLIMpus.inputs_LIM import *
from scipy.special import gammaln

# Number of terms in the series. The terms peak at n ~ sigma_x^2 and fall off
# factorially; 260 is far beyond convergence for sigma_PS <= 6 (sigma_x^2 <= 18).
_NMAX_SERIES = 260


# --------------------------------------------------------------------------- #
# The burstiness amplitude
# --------------------------------------------------------------------------- #

def sigma_PS_of_M(massVector, LineParams):
    """M26 mass-dependent rms log-SFR scatter sigma_PS(Mh).

    Parameters
    ----------
    massVector : array
        Halo masses in Msun.
    LineParams : Line_Parameters
        Uses sigPS_piv_bursty, log10M_piv_bursty, dsigPS_dlog10M_bursty and the
        clamps sigPS_min_bursty / sigPS_max_bursty.

    Returns
    -------
    array, same shape as massVector.
    """
    sigPS = (LineParams.sigPS_piv_bursty
             + LineParams.dsigPS_dlog10M_bursty
             * (np.log10(massVector) - LineParams.log10M_piv_bursty))

    sigPS = np.maximum(sigPS, LineParams.sigPS_min_bursty)
    if LineParams.sigPS_max_bursty is not None:
        sigPS = np.minimum(sigPS, LineParams.sigPS_max_bursty)

    return sigPS


def sigma_x2_of_M(massVector, LineParams):
    "Variance of x = ln(SFR) per direction: sigma_x^2 = sigma_PS^2 / 2."
    return 0.5 * sigma_PS_of_M(massVector, LineParams)**2


# --------------------------------------------------------------------------- #
# The luminosity variance, in closed form
# --------------------------------------------------------------------------- #

def _series_coefficients(sigma_x2, nmax=_NMAX_SERIES):
    "exp(n log sigma_x^2 - lngamma(n+1)) for n = 1..nmax, shape (..., nmax)."
    n = np.arange(1, nmax + 1)
    sx2 = np.asarray(sigma_x2, dtype=float)[..., np.newaxis]
    return np.exp(n * np.log(sx2) - gammaln(n + 1)), n


def _I_ramp(T, a):
    "Int_0^T (T-u) exp(-a u) du"
    return T / a - (1.0 - np.exp(-a * T)) / a**2


def _I_flat(A, a):
    "Int_0^A exp(-a u) du"
    return (1.0 - np.exp(-a * A)) / a


def V_lambda(massVector, LineParams, t_Myr=None):
    """Dimensionless luminosity variance Var(L)/Lbar^2 at fixed halo mass.

    Eq. 5 of 2605.13967, for a top-hat SPS window of width t_lambda.

    Parameters
    ----------
    massVector : array
        Halo masses in Msun.
    LineParams : Line_Parameters
    t_Myr : float, optional
        Override the line's effective window width t_lambda.

    Returns
    -------
    array, same shape as massVector.
    """
    t = LineParams.t_Myr_per_line if t_Myr is None else t_Myr
    c, n = _series_coefficients(sigma_x2_of_M(massVector, LineParams))
    a = n / LineParams.tauPS_Myr_bursty

    return 2.0 * np.sum(c * _I_ramp(t, a), axis=-1) / t**2


def V_12(massVector, LineParams, LineParams_cross):
    """Same-halo cross moment cov(L1, L2)/(Lbar1 Lbar2) at fixed halo mass.

    Eqs. 8-9 of 2605.13967: the two top-hat windows of widths t1, t2 see the
    same OU field, and the geometric weight is the length of the diagonal slice
    u = t_1 - t_2 inside the rectangle [0,t1] x [0,t2]. Reduces exactly to
    V_lambda when t1 == t2.

    tau_PS is a property of the star-formation process, not of the line, so the
    two Line_Parameters must agree on it.
    """
    if LineParams.tauPS_Myr_bursty != LineParams_cross.tauPS_Myr_bursty:
        raise ValueError('tauPS_Myr_bursty differs between the two lines; it is a '
                         'property of the SFR process, not of the line.')

    t1, t2 = LineParams.t_Myr_per_line, LineParams_cross.t_Myr_per_line
    tlo, thi = (t1, t2) if t1 <= t2 else (t2, t1)

    c, n = _series_coefficients(sigma_x2_of_M(massVector, LineParams))
    a = n / LineParams.tauPS_Myr_bursty

    A = thi - tlo
    weight = _I_ramp(tlo, a) + tlo * _I_flat(A, a) + np.exp(-a * A) * _I_ramp(tlo, a)

    return np.sum(c * weight, axis=-1) / (t1 * t2)


def V_lambda_exponential(massVector, LineParams, t_e_Myr):
    """Same as V_lambda but for an exponential Green's function G(t) = e^{-t/t_e}/t_e.

    Appendix B of 2605.13967. The autocorrelation of an exponential kernel is
    itself exponential, so Var(L)/Lbar^2 = (1/t_e) Int_0^inf du e^{-u/t_e}
    [exp(sigma_x^2 e^{-u/tau}) - 1], which the same series turns into
    Sum_n (sigma_x^2)^n/n! * 1/(1 + n t_e/tau).

    Used to check the top-hat approximation: the paper calibrates t_e = t_lambda/2.
    """
    c, n = _series_coefficients(sigma_x2_of_M(massVector, LineParams))

    return np.sum(c / (1.0 + n * t_e_Myr / LineParams.tauPS_Myr_bursty), axis=-1)


# --------------------------------------------------------------------------- #
# Boosts
# --------------------------------------------------------------------------- #

def _extra_scatter_factor(LineParams, LineParams_cross=None):
    """exp(sigma_extra^2) in natural log, for scatter that is NOT burstiness.

    The Discussion of 2605.13967 lists dust attenuation, geometry, central-satellite
    splits and Lyman-alpha radiative transfer as additional sources of scatter in
    L at fixed Mh, and quotes a ~20-30% extra enhancement for 0.1-0.2 dex. That
    number corresponds to the FULLY TIME-CORRELATED limit, in which the extra
    scatter multiplies the second moment by exp(sigma_extra^2) independently of
    the SPS window -- which is what is implemented here. Treating it instead as
    part of sigma_x^2 inside the OU correlator would give a smaller,
    window-dependent enhancement; if that is what you want, raise sigma_PS.
    """
    s1 = LineParams.sigma_extra_dex * np.log(10.)
    s2 = s1 if LineParams_cross is None else LineParams_cross.sigma_extra_dex * np.log(10.)

    return np.exp(s1 * s2)


def boost_per_halo(massVector, LineParams, t_Myr=None):
    """Per-halo shot-noise boost <L^2>/Lbar^2 = exp(sigma_xi^2) = 1 + V_lambda.

    This is what Fig. 1 of 2605.13967 plots.
    """
    return (1.0 + V_lambda(massVector, LineParams, t_Myr=t_Myr)) \
        * _extra_scatter_factor(LineParams)


def boost_per_halo_cross(massVector, LineParams, LineParams_cross):
    "Per-halo cross boost <L1 L2>/(Lbar1 Lbar2) = 1 + V_12 (Eq. 10)."
    return (1.0 + V_12(massVector, LineParams, LineParams_cross)) \
        * _extra_scatter_factor(LineParams, LineParams_cross)


def sigma_xi2(V):
    "Effective log-variance sigma_xi^2 = ln(1 + V), Eq. 6."
    return np.log1p(V)


def sigma_L_equivalent_dex(boost):
    """The mass-independent lognormal scatter, in dex, that would give the same boost.

    Standard LIM forecasts adopt a phenomenological sigma_L ~ 0.3-0.5 dex, which
    produces a boost exp((sigma_L ln10)^2). Inverting that puts the burstiness
    prediction on the same axis; it is the comparison Fig. 2 of 2605.13967 makes.
    """
    return np.sqrt(np.log(boost)) / np.log(10.)


def mass_average(quantity, weight, logtabMh):
    """Weighted average over d log Mh, along the last axis.

    Parameters
    ----------
    quantity : array, (..., nM)
    weight : array, (..., nM)
        For a shot-noise average this is dn/dlogMh * Lbar1 * Lbar2.
    logtabMh : array, (nM,)
    """
    num = np.trapezoid(weight * quantity, logtabMh, axis=-1)
    den = np.trapezoid(weight, logtabMh, axis=-1)

    return num / den


def R_cross(V1, V2, V12, weight1, weight2, weight12, logtabMh):
    """Cross-correlation coefficient of the bursty shot-noise EXCESS, Eq. 11.

    With Delta_ij = P_shot^ij / P_shot^ij|det - 1 = <V_ij> weighted by
    dn/dlogMh Lbar_i Lbar_j,

        R_12 = Delta_12 / sqrt(Delta_1 Delta_2).

    The deterministic (halo-discreteness) term and the luminosity normalisations
    divide out, leaving only the burstiness-induced part: R depends mainly on
    tau_PS and only weakly on sigma_PS, which is what makes it a clean probe of
    the burst coherence time.
    """
    d1 = mass_average(V1, weight1, logtabMh)
    d2 = mass_average(V2, weight2, logtabMh)
    d12 = mass_average(V12, weight12, logtabMh)

    return d12 / np.sqrt(d1 * d2)
