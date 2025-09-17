from study_anticorr import * 
from oLIMpus.analysis import * 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import binned_statistic


_dlogzint_target_low = 0.02/(UP.precisionboost*3)
_dlogzint_target = 0.02/UP.precisionboost
zmin_low = ZMIN+1.
Nzintegral_low= np.ceil(1.0 + np.log(zmin_low/ZMIN)/_dlogzint_target_low).astype(int)
Nzintegral = np.ceil(1.0 + np.log(constants.ZMAX_INTEGRAL/zmin_low)/_dlogzint_target).astype(int)
dlogzint_low = np.log(zmin_low/ZMIN)/(Nzintegral_low-1.0) #exact value rather than input target above
dlogzint = np.log(constants.ZMAX_INTEGRAL/ZMIN)/(Nzintegral-1.0) #exact value rather than input target above
zvals_low = np.logspace(np.log10(ZMIN), np.log10(zmin_low), Nzintegral_low)
zvals_high = np.logspace(np.log10(zmin_low), np.log10(constants.ZMAX_INTEGRAL), Nzintegral)
zvals = np.concatenate((zvals_low,zvals_high[1:]))

#zvals_maps = (np.linspace(16, 6, int((16 - 6) / 0.1) + 1))


plot_path = path + 'plots/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path, exist_ok=True)

xmin_plot = 5.5
xmax_plot = 18.5

xHmin = 0.01
xHmax = 0.1

color_list=[a.colors[2],a.colors[6]]#a.colors[4]


def max_estimator(model,which_par,par_vals,Lbox,with_shotnoise=True,Nbox=None,save_fig=False):

    outputs = import_model(model,which_par,par_vals,Lbox,with_shotnoise,Nbox)
    p = outputs['p']
    xHv = outputs['xHv']

    Nbin = [0,10,20]
    lsN = ['-','--',':']
    par_vals_red = [par_vals[0],par_vals[-1]] if which_par != 'fesc' else [par_vals[3],par_vals[-3]] 

    use_zvals = zvals
    use_par_vals = par_vals 
    fig, ax = plt.subplots(1,3,figsize=(17,5))
    for n in Nbin:
        if n > 0:
            zs, ps = binned_Pearson(n, p, par_vals,use_zvals)
        else:
            zs = use_zvals
            ps = p 

        max_est = np.zeros(len(use_par_vals))
        z_max_est = np.zeros(len(use_par_vals))
        xH_max_est = np.zeros(len(use_par_vals))
        z_xH3pc = np.zeros(len(use_par_vals))

        par_name = r'$\epsilon_*$' if which_par == 'fstar' else r'$f_{\rm esc}$' if which_par ==  'fesc' else r'$\alpha_*$' if which_par == 'alphastar' else r'$\beta_*$' if which_par == 'betastar' else r'$\log_{10}L_X/{\rm SFR}$' if which_par == 'LX' else ''

        for ii in range(len(par_vals_red)):
            
            i = list(par_vals).index(par_vals_red[ii])
            zxHz = interp1d(xHv[i], use_zvals,bounds_error=False, fill_value=0.)

            if n == 0:
                ax[0].plot(zs,ps[i],color=color_list[ii], label=par_name + r'$=%g$'%round(par_vals[i],3),ls=lsN[Nbin.index(n)])
                ax[0].axvline(zxHz(1-0.03),label=r'$z(x_{\rm H}=3\%)$',color=color_list[ii])
            else:
                ax[0].plot(zs,ps[i],color=color_list[ii],ls=lsN[Nbin.index(n)])

        for i in range(len(use_par_vals)):
            
            ps[i][np.isnan(ps[i])] = 0.
            max_est[i] = np.max(ps[i])

            zp_interp = interp1d(ps[i], zs, bounds_error=False, fill_value=0.)

            z_max_est[i] = zp_interp(max_est[i])

            xHz = interp1d(use_zvals,xHv[i], bounds_error=False, fill_value=0.)
            zxHz = interp1d(xHv[i], use_zvals,bounds_error=False, fill_value=0.)

            xH_max_est[i] = (1-xHz(z_max_est[i]))*100.
            z_xH3pc[i] = zxHz(1-0.03)

        if n == 0:
            labelN = r'$\rm Full$'
        else:
            labelN = r'$N_{\rm bin}=%g$'%n

        ax[1].plot(use_par_vals,z_max_est,ls=lsN[Nbin.index(n)],color=color_list[0],label=labelN)
        if n == 0 :
           ax[1].plot(use_par_vals,z_xH3pc,ls=lsN[Nbin.index(n)],color=colors[0],label=r'$z(x_{\rm H}=3\%)$')
        ax[2].plot(use_par_vals,xH_max_est,ls=lsN[Nbin.index(n)],label=labelN,color=colors[0])

    ax[0].set_ylabel(r'$P(z)$')
    ax[1].set_ylabel(r'$z_{\rm max}$')
    ax[2].set_ylabel(r'$\bar{x}_{\rm H}(z_{\rm max})\,(\%)$')

    ax[0].set_xlabel(r'$z$')
    ax[1].set_xlabel(par_name)
    ax[2].set_xlabel(par_name)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.tight_layout()
    if save_fig:
        plt.savefig(plot_path + 'estimator_' + model + '_' + which_par + '_' + str(Lbox) + '.png',dpi=300,bbox_inches='tight')

    plt.show()

    return 


def binned_Pearson(Nbin, p_input, par_vals, use_z_input, z_edges = None):

    if use_z_input[-1] < use_z_input[0]:
        use_z = use_z_input[::-1]
    else:
        use_z = use_z_input

    if z_edges is None:
        z_edges = np.logspace(np.log10(use_z[0]), np.log10(use_z[-1]), Nbin+1)   # N+1 edges → N bins

    # Create step function
    z_steplike = np.repeat(z_edges, 2)[1:-1]  # repeat each edge to get flat steps

    p_steplike = []
    for i in range(len(par_vals)):
        # --- bin p (take the mean in each bin; change 'mean' to 'sum', 'median', etc. if needed) ---

        if use_z_input[-1] < use_z_input[0]:
            p = p_input[i][::-1]
        else:
            p = p_input[i]

        p[np.isnan(p[i])] = 0.

        p_binned, _, _ = binned_statistic(use_z, p, statistic='mean', bins=z_edges)

        p_steplike.append(np.repeat(p_binned, 2)) # repeat each value to hold it flat

    return z_steplike, p_steplike


def plot_onepoint(model,which_par,par_vals_all,Lbox,with_shotnoise=True,Nbox=None,save_fig=False):

    par_name = r'$\epsilon_*$' if which_par == 'fstar' else r'$f_{\rm esc}$' if which_par ==  'fesc' else r'$\alpha_*$' if which_par == 'alphastar' else r'$\beta_*$' if which_par == 'betastar' else r'$\log_{10}\frac{L_X}{\rm SFR}$' if which_par=='LX' else ''

    # first plot the fiducial
    outputs = import_model(model,'fiducial',[0],Lbox,with_shotnoise,Nbox)

    p = outputs['p']
    T21 = outputs['T21']
    xHv = outputs['xHv']
    r = np.asarray(outputs['r'])
    k_cross = outputs['k_cross']

    ks = [0.2,0.5]
    ls = ['-','--']

    r[0][np.isnan(r[0])] = 0.
    rk_int = RegularGridInterpolator((zvals, k_cross[0][0]), r[0], bounds_error=False, fill_value=0.)

    fig, ax = plt.subplots(1,4,figsize=(23,4.5))

    if Lbox == 600:
        use_zvals = zvals
    else:
        use_zvals = zvals

    zxH = interp1d(xHv[0], use_zvals, bounds_error=False, fill_value=0.)

    p[0][np.isnan(p[0])] = 0.

    par_name_def = 'epsstar' if which_par == 'fstar' else 'fesc10' if which_par == 'fesc' else 'L40_xray' if which_par == 'LX' else which_par
    val = np.log10(1e40*AstroParams_input_fid_use[par_name_def]) if which_par=='LX' else AstroParams_input_fid_use[par_name_def]
    
    fiducial_line, = ax[0].plot(use_zvals,1-xHv[0],color=colors[0],label=par_name + r'$=%g$'%round(val,3))

    ax[1].plot(zvals,T21[0],color=colors[0],label=par_name + r'$=%g$'%round(val,3))

    ax[2].plot(use_zvals,p[0],color=colors[0],label=par_name + r'$=%g$'%round(val,3))

    for k in range(len(ks)):
        points = np.column_stack([zvals, np.full_like(zvals, ks[k])])
        ax[3].plot(use_zvals,rk_int(points),color=colors[0],ls=ls[k],label= r'$k=%g\,{\rm Mpc}^{-1}$'%ks[k])

    other_lines = []

    for j in range(len(ax)):
        # ax[j].axvline(zxH(1-xHmin),linewidth=1.7,color=colors[0],alpha=0.5,ls='--')
        # ax[j].axvline(zxH(1-xHmax),linewidth=1.7,color=colors[0],alpha=0.5,ls='-')
        ax[j].axvspan(zxH(1-xHmin), zxH(1-xHmax),color=colors[0], alpha=0.2)

        ax[j].axhline(0.,linewidth=0.5,color=colors[0])
        ax[j].set_xlim(xmin_plot,xmax_plot)
        ax[j].set_xlabel(r'$z$')


    if which_par != 'fiducial':
        outputs = import_model(model,which_par,par_vals_all,Lbox,with_shotnoise,Nbox)

        p = outputs['p']
        T21 = outputs['T21']
        xHv = outputs['xHv']
        r = np.asarray(outputs['r'])
        k_cross = outputs['k_cross']
        
        par_vals = [par_vals_all[0],par_vals_all[-1]] #par_vals_all[int(len(par_vals_all)/2)]

        for ii in range(len(par_vals)):

            i = list(par_vals_all).index(par_vals[ii])

            r[i][np.isnan(r[i])] = 0.
            rk_int = RegularGridInterpolator((zvals, k_cross[i][0]), r[i], bounds_error=False, fill_value=0.)

            zxH = interp1d(xHv[i], use_zvals, bounds_error=False, fill_value=0.)

            p[i][np.isnan(p[i])] = 0.

            line, = ax[0].plot(use_zvals,1-xHv[i],color=color_list[ii], label=par_name + r'$=%g$'%round(par_vals[ii],3))
            other_lines.append(line)
            ax[1].plot(zvals,T21[i],color=color_list[ii], label=par_name + r'$=%g$'%round(par_vals[ii],3))

            ax[2].plot(use_zvals,p[i],color=color_list[ii], label=par_name + r'$=%g$'%round(par_vals[ii],3))

            for k in range(len(ks)):
                points = np.column_stack([zvals, np.full_like(zvals, ks[k])])
                ax[3].plot(use_zvals,rk_int(points),color=color_list[ii],ls=ls[k],)

            for j in range(len(ax)):
                # ax[j].axvline(zxH(1-xHmin),linewidth=1.7,color=color_list[ii],alpha=0.5,ls='--')
                # ax[j].axvline(zxH(1-xHmax),linewidth=1.7,color=color_list[ii],alpha=0.5,ls='-')
                if which_par !='LX':
                    ax[j].axvspan(zxH(1-xHmin), zxH(1-xHmax),color=color_list[ii], alpha=0.2)
                
                ax[j].axhline(0.,linewidth=0.5,color=colors[0])
                ax[j].set_xlim(xmin_plot,xmax_plot)
                ax[j].set_xlabel(r'$z$')

    ax[2].set_ylim(-1,1)
    ax[3].set_ylim(-1.3,1.3)

    ax[0].set_ylabel(r'$\bar{x}_{\rm H}(z)$')
    ax[1].set_ylabel(r'$\bar{T}_{21}(z)$')
    ax[2].set_ylabel(r'$P(z)$')
    ax[3].set_ylabel(r'$r_{\rm cross}(z)$')

    ax[3].legend(loc=2,ncol=2,fontsize=13,columnspacing=0.5)

    other_lines.insert(1,fiducial_line)
    # Get handles and labels from one axis (they're the same across all)
    handles = other_lines
    labels = [h.get_label() for h in handles]

    extra_handles = [
        Patch(facecolor=colors[0], alpha=0.5, label=r'$x_{\rm H}\in \[1\%,10\,\]$'),
        # Line2D([0], [0], color=colors[0], linewidth=1.7, alpha=0.5,linestyle='--', label=r'$\bar{x}_{\rm H}=%g$'%(100*xHmin)),
        # Line2D([0], [0], color=colors[0], linewidth=1.7, alpha=0.5,linestyle='-', label=r'$\bar{x}_{\rm H}=%g$'%(100*xHmax)),
    ]
    # extra_labels =[r'$\bar{x}_{\rm H}=%g$'%(100*xHmin)+r'$\%$',r'$\bar{x}_{\rm H}=%g$'%(100*xHmax)+r'$\%$']
    extra_labels = [r'$x_{\rm H}\in [1\%,10\%]$']

    all_handles = handles + extra_handles
    all_labels = labels + extra_labels
    
    # Add a single legend to the right
    fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(1.0, 0.5), frameon=False)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0., 0.5])
    if save_fig:
        SNlabel = '_noSN' if not with_shotnoise else ''
        plt.savefig(plot_path + model + '_' + which_par + '_' + str(Lbox) + '_bar' + SNlabel + '.png',dpi=300,bbox_inches='tight')

    plt.show()

    return 


def plot_SN(model,Lbox,Nbox=None,save_fig=False):

    with_shotnoise= [True,False]
    labels = [r'$\rm With\,SN$',r'$\rm Without\,SN$']
    line = ['-','--']

    Nbin = [0]#,10,20]
    alpha = [1,0.6,0.4]
    fig, ax = plt.subplots(1,3,figsize=(17,5))
    for jj in range(len(with_shotnoise)):

        # first plot the fiducial
        outputs = import_model(model,'fiducial',[0],Lbox,with_shotnoise[jj],Nbox)

        p = outputs['p']
        T21 = outputs['T21']
        xHv = outputs['xHv']

        use_zvals = zvals

        zxH = interp1d(xHv[0], use_zvals, bounds_error=False, fill_value=0.)

        p[0][np.isnan(p[0])] = 0.


        ax[0].plot(use_zvals,1-xHv[0],color=colors[0],label=labels[jj],ls=line[jj])

        ax[1].plot(zvals,T21[0],color=colors[0],label=labels[jj],ls=line[jj])

        idv = 0
        if with_shotnoise[jj]:
            for n in Nbin:
                if n > 0:
                    use_zvals1, ps = binned_Pearson(n, p, [0],use_zvals)
                else:
                    use_zvals1 = zvals
                    ps = p 

                ax[2].plot(use_zvals1[idv:],ps[0][idv:],color=colors[0],label=labels[jj],ls=line[jj],alpha=alpha[Nbin.index(n)])
        else:
            ax[2].plot(use_zvals[idv:],p[0][idv:],color=colors[0],label=labels[jj],ls=line[jj],)

        for j in range(len(ax)):
            # ax[j].axvline(zxH(1-0.01),linewidth=1.7,color=colors[0],alpha=0.5,ls='--')
            # ax[j].axvline(zxH(1-0.05),linewidth=1.7,color=colors[0],alpha=0.5,ls='-')
            # ax[j].axvline(zxH(1-0.15),linewidth=1.7,color=colors[0],alpha=0.5,ls=':')

            ax[j].axvspan(zxH(1-xHmin), zxH(1-xHmax),color=colors[0], alpha=0.2)

            ax[j].axhline(0.,linewidth=0.5,color=colors[0])
            ax[j].set_xlim(xmin_plot,xmax_plot)
            ax[j].set_xlabel(r'$z$')


    ax[2].set_ylim(-1,1)

    ax[0].set_ylabel(r'$\bar{x}_{\rm H}(z)$')
    ax[1].set_ylabel(r'$\bar{T}_{21}(z)$')
    ax[2].set_ylabel(r'$P(z)$')

    # Get handles and labels from one axis (they're the same across all)
    handles, labels = ax[0].get_legend_handles_labels()

    extra_handles = [
        Patch(facecolor=colors[0], alpha=0.5, label=r'$x_{\rm H}\in \[1\%,10\,\]$'),
        # Line2D([0], [0], color=colors[0], linewidth=1.7, alpha=0.5,linestyle='--', label=r'$\bar{x}_{\rm H}=%g$'%(100*xHmin)),
        # Line2D([0], [0], color=colors[0], linewidth=1.7, alpha=0.5,linestyle='-', label=r'$\bar{x}_{\rm H}=%g$'%(100*xHmax)),
    ]
    # extra_labels =[r'$\bar{x}_{\rm H}=%g$'%(100*xHmin)+r'$\%$',r'$\bar{x}_{\rm H}=%g$'%(100*xHmax)+r'$\%$']
    extra_labels = [r'$x_{\rm H}\in [1\%,10\%]$']

    all_handles = handles + extra_handles
    all_labels = labels + extra_labels
    
    # Add a single legend to the right
    fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(1.01, 0.6), frameon=False)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    if save_fig:
        plt.savefig(plot_path + model + '_SN_' + str(Lbox) + '_'+ str(Nbox) + '.png',dpi=300,bbox_inches='tight')

    plt.show()

    return 


def plot_models(Lbox,Nbox=None,with_shotnoise=True,save_fig=False):

    models = ['SFRD', 'OIII', 'CO21',  'CII', 'Ha']
    labels = [r'$\rm SFRD$', r'$\rm O_{III}$', r'$\rm CO_{21}$', r'$\rm C_{II}$', r'$\rm H\alpha$']

    Nbin = [0]#,10,20]

    kv = [0.1,0.5]
    ls= ['-','--']
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    for model in models:

        # first plot the fiducial
        outputs = import_model(model,'fiducial',[0],Lbox,with_shotnoise,Nbox)

        p = outputs['p']
        k_cross = outputs['k_cross']
        r = outputs['r']
        T21 = outputs['T21']
        xHv = outputs['xHv']

        use_zvals = zvals

        zxH = interp1d(xHv[0], use_zvals, bounds_error=False, fill_value=0.)

        p[0][np.isnan(p[0])] = 0.

        jj = models.index(model)
        if jj == 0:
            ax[0].plot(use_zvals,1-xHv[0],label=labels[jj])
            ax[1].plot(zvals,T21[0],label=labels[jj])

        idv = 0
        ax[2].plot(use_zvals[idv:],p[0][idv:],label=labels[jj],color=colors[jj*2])

        rk_int = RegularGridInterpolator((use_zvals, k_cross[0][0]), r[0], bounds_error=False, fill_value=0.)
        for k in range(len(kv)):
            points = np.column_stack([zvals, np.full_like(zvals, kv[k])])

            if jj == 0:
                ax[3].plot(use_zvals[idv:],rk_int(points),label=r'$k=%g\,{\rm Mpc}^{-1}$'%kv[k],color=colors[jj*2],ls=ls[k])
            else:
                ax[3].plot(use_zvals[idv:],rk_int(points),color=colors[jj*2],ls=ls[k])

        for j in range(len(ax)):

            if jj == 0 :
                ax[j].axvspan(zxH(1-xHmin), zxH(1-xHmax),color=colors[0], alpha=0.2)

                ax[j].axhline(0.,linewidth=0.5,color=colors[0])
                ax[j].set_xlim(xmin_plot,xmax_plot)
                ax[j].set_xlabel(r'$z$')


        ax[2].set_ylim(-1,1)
        ax[3].set_ylim(-1,1)
        ax[3].legend()

        ax[0].set_ylabel(r'$\bar{x}_{\rm H}(z)$')
        ax[1].set_ylabel(r'$\bar{T}_{21}(z)$')
        ax[2].set_ylabel(r'$P(z)$')
        ax[3].set_ylabel(r'$r(z)$')

    # Get handles and labels from one axis (they're the same across all)
    handles, labels = ax[2].get_legend_handles_labels()

    extra_handles = [
        Patch(facecolor=colors[0], alpha=0.2, label=r'$x_{\rm H}\in \[1\%,10\,\]$'),
        # Line2D([0], [0], color=colors[0], linewidth=1.7, alpha=0.5,linestyle='--', label=r'$\bar{x}_{\rm H}=%g$'%(100*xHmin)),
        # Line2D([0], [0], color=colors[0], linewidth=1.7, alpha=0.5,linestyle='-', label=r'$\bar{x}_{\rm H}=%g$'%(100*xHmax)),
    ]
    # extra_labels =[r'$\bar{x}_{\rm H}=%g$'%(100*xHmin)+r'$\%$',r'$\bar{x}_{\rm H}=%g$'%(100*xHmax)+r'$\%$']
    extra_labels = [r'$x_{\rm H}\in [1\%,10\%]$']

    all_handles = handles + extra_handles
    all_labels = labels + extra_labels
    
    # Add a single legend to the right
    fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(1.01, 0.6), frameon=False)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    if save_fig:
        plt.savefig(plot_path + 'models_SN_' + str(Lbox) + '_'+ str(Nbox) + '.png',dpi=300,bbox_inches='tight')

    plt.show()

    return 
