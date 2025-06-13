import numpy as np 
from oLIMpus import inputs_LIM

########################################################
### UV AND OPTICAL 
########################################################

# from arXiv:2409.03997
def Yang24(line, dotM):

    if line == 'OIII':
        line_dict = inputs_LIM.Yang24_OIII_params
    elif line == 'OII':
        line_dict = inputs_LIM.Yang24_OII_params
    elif line== 'Ha':
        line_dict = inputs_LIM.Yang24_Ha_params
    elif line == 'Hb':
        line_dict = inputs_LIM.Yang24_Hb_params
    else:
        print('\nLINE NOT IMPLEMENTED YET IN YANG24')
        return -1

    alpha = line_dict['alpha']
    beta = line_dict['beta']
    N = line_dict['N']
    SFR1 = line_dict['SFR1']

    L_line = 2. * N * dotM / ((dotM / SFR1)**(-alpha) + (dotM / SFR1)**beta)

    log10_L = np.log10(L_line)
    
    return log10_L


# from arXiv:2111.02411
def THESAN21(line, dotM):
    if line == 'OIII':
        line_dict = inputs_LIM.THESAN21_OIII_params
    elif line == 'OII':
        line_dict = inputs_LIM.THESAN21_OII_params
    elif line == 'Ha':
        line_dict = inputs_LIM.THESAN21_Ha_params
    elif line == 'Hb':
        line_dict = inputs_LIM.THESAN21_Hb_params
    else:
        print('\nLINE NOT IMPLEMENTED YET IN THESAN21')
        return -1

    a = line_dict['a']
    ma = line_dict['ma']
    mb = line_dict['mb']
    log10_SFR_b = line_dict['log10_SFR_b']
    mc = line_dict['mc']
    log10_SFR_c = line_dict['log10_SFR_c']

    log10_SFR = np.log10(dotM)

    log10_L = np.empty_like(log10_SFR)

    cond1 = log10_SFR < log10_SFR_b
    cond2 = (log10_SFR >= log10_SFR_b) & (log10_SFR < log10_SFR_c)
    cond3 = log10_SFR >= log10_SFR_c

    log10_L[cond1] = a + ma * log10_SFR[cond1]
    log10_L[cond2] = a + (ma - mb) * log10_SFR_b + mb * log10_SFR[cond2]
    log10_L[cond3] = a + (ma - mb) * log10_SFR_b + (mb - mc) * log10_SFR_c + mc * log10_SFR[cond3]

    return log10_L

########################################################
### INFRARED CII
########################################################

# from arXiv:1711.00798
def Lagache18(line, dotM, z):

    if line != 'CII':
        print('\nLINE NOT IMPLEMENTED YET IN LAGACHE18')
        return -1

    line_dict = inputs_LIM.Lagache18_CII_params

    alpha_SFR =line_dict['alpha_SFR_0'] + line_dict['alpha_SFR'] * z

    beta_SFR = line_dict['beta_SFR_0'] + line_dict['beta_SFR'] * z

    try:
        alpha_SFR[alpha_SFR < 0.] = 0. 
    except:
        if alpha_SFR < 0.:
            alpha_SFR = 0.

    log10_L = alpha_SFR * np.log10(dotM) + beta_SFR     

    return log10_L


########################################################
### SUB-MM CO
########################################################

# from arXiv:2108.07716
def Yang21(line, massVector, z):

    YangEmp_f2 = lambda x1, x2, x3, zz: 1 + x2*z + x3*zz**2
    YangEmp_f1 = lambda x1, x2, x3, zz: x1*np.exp(-zz/x2) + x3

    if line == 'CO21':
        line_dict = inputs_LIM.Yang21_CO21_params
                    
        logM1 = np.where(z < 4.0,
                        YangEmp_f2(12.12, -0.1704, 0, z),
                        np.where(z < 5.0,
                                YangEmp_f2(11.74, -0.07050, 0, z),
                                YangEmp_f2(11.63, -0.05266, 0, z))) # !!! note that this above 8.5 is not correct

        logN = np.where(z < 4.0,
                        YangEmp_f2(-5.95, 0.278, -0.0521, z),
                        np.where(z < 5.0,
                                YangEmp_f2(-5.57, -0.025, 0, z),
                                YangEmp_f2(-5.26, -0.0849, 0, z)))# !!! note that this above 8.5 is not correct

        a = np.where(z < 4.0,
                    YangEmp_f2(1.69, 0.126, -0.028, z),
                    np.where(z < 5.0,
                            YangEmp_f2(4.557, -1.215, 0.13, z),
                            YangEmp_f2(2.47, -0.210, 0.0132, z)))# !!! note that this above 8.5 is not correct

        b = np.where(z < 4.0,
                    YangEmp_f1(1.8, 2.76, -0.0678, z),
                    np.where(z < 5.0,
                            YangEmp_f2(0.657, -0.0794, 0, z),
                            YangEmp_f1(38.3, 0.841, 0.169, z)))# !!! note that this above 8.5 is not correct
        
    elif line == 'CO10':
        logM1 = np.where(z < 4.0,
                        YangEmp_f2(12.13, -0.1678, 0, z),
                        np.where(z < 5.0,
                                YangEmp_f2(11.75, -0.06833, 0, z),
                                YangEmp_f2(11.63, -0.04943, 0, z)))# !!! note that this above 8.5 is not correct

        logN = np.where(z < 4.0,
                        YangEmp_f2(-6.855, 0.2366, -0.05013, z),
                        np.where(z < 5.0,
                                YangEmp_f2(-6.554, -0.03725, 0, z),
                                YangEmp_f2(-6.274, -0.09087, 0, z)))# !!! note that this above 8.5 is not correct

        a = np.where(z < 4.0,
                    YangEmp_f2(1.642, 0.1663, -0.03238, z),
                    np.where(z < 5.0,
                            YangEmp_f2(3.73, -0.833, 0.0884, z),
                            YangEmp_f2(2.56, -0.223, 0.0142, z)))# !!! note that this above 8.5 is not correct

        b = np.where(z < 4.0,
                    YangEmp_f1(1.77, 2.72, -0.0827, z),
                    np.where(z < 5.0,
                            YangEmp_f2(0.598, -0.0710, 0, z),
                            YangEmp_f1(33.4, 0.846, 0.16, z)))# !!! note that this above 8.5 is not correct

    else:
        print('\nLINE NOT IMPLEMENTED YET IN YANG21')
        return -1

    # Empirically fit parameter values for the Yang+ empirical XX model
    A = line_dict['A']
    
    M1 = 10**logM1
    N = 10**logN

    log10_L = np.log10(A*(2*N*massVector/((massVector/M1)**-a+(massVector/M1)**b)))

    return log10_L


# from arXiv:1503.08833
def Li16(line, dotM):

    if line == 'CO21':
        line_dict = inputs_LIM.Li16_C021_params
    else:
        print('\nLINE NOT IMPLEMENTED YET IN LI16')
        return -1

    alpha = line_dict['alpha']
    beta = line_dict['beta']
    dMF = line_dict['dMF']
    L0 = line_dict['L0']

    log10_SFR = np.log10(dotM)

    L_IR = 10**log10_SFR / (dMF*1e-10)
    Lprime = (10.**-beta * L_IR)**(1./alpha)

    log10_L = np.log10(L0*Lprime)

    return log10_L