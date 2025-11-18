
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np 
import pickle 
from scipy.interpolate import interp1d, RegularGridInterpolator
import os
import matplotlib.pyplot as plt

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


alphastar_fid = AstroParams_input_fid_use['alphastar']
betastar_fid = AstroParams_input_fid_use['betastar']
epsstar_fid = AstroParams_input_fid_use['epsstar']
fesc_fid = AstroParams_input_fid_use['fesc10']
LX_fid = np.log10(AstroParams_input_fid_use['L40_xray'] * 1e40)

path = '../run_random_parameters/'

def import_random_model(nid, model,Lbox,with_shotnoise=True,Nbox = None, _R=None,include_partlion=True):

    save_path = path + str(nid) + '_random_' + model + '_' + str(Lbox) + '_' + str(Nbox) 

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

def crossing_brackets(z, p, target=0.0, tol=1e-12):
    """Return list of (z[i], z[i+1]) intervals where p crosses target (sign change)."""
    z = np.asarray(z)
    p = np.asarray(p)
    s = p - target
    # detect sign changes ignoring exact-equals (plateaus handled separately)
    sign_changes = np.where((s[:-1] * s[1:]) < 0)[0]
    brackets = [(float(z[i]), float(z[i+1])) for i in sign_changes]
    return brackets

use_zvals = np.asarray([ 5.        ,  5.03266365,  5.06554068,  5.09863249,  5.13194048,
        5.16546606,  5.19921066,  5.2331757 ,  5.26736262,  5.30177288,
        5.33640793,  5.37126924,  5.40635829,  5.44167657,  5.47722558,
        5.51300681,  5.5490218 ,  5.58527206,  5.62175913,  5.65848457,
        5.69544992,  5.73265675,  5.77010665,  5.8078012 ,  5.845742  ,
        5.88393065,  5.92236878,  5.96105802,  6.        ,  6.12007941,
        6.24256199,  6.36749585,  6.49493003,  6.62491459,  6.75750056,
        6.89274   ,  7.03068603,  7.17139279,  7.31491556,  7.46131068,
        7.61063564,  7.76294907,  7.91831079,  8.0767818 ,  8.23842433,
        8.40330185,  8.5714791 ,  8.74302212,  8.91799827,  9.09647626,
        9.27852617,  9.46421949,  9.65362914,  9.84682948, 10.04389639,
       10.24490724, 10.44994097, 10.65907809, 10.87240072, 11.08999262,
       11.31193925, 11.53832774, 11.769247  , 12.0047877 , 12.24504233,
       12.49010523, 12.74007263, 12.99504269, 13.25511553, 13.52039326,
       13.79098006, 14.06698218, 14.34850799, 14.63566805, 14.9285751 ,
       15.22734418, 15.53209259, 15.84294   , 16.16000847, 16.48342251,
       16.81330911, 17.1497978 , 17.49302073, 17.84311265, 18.20021105,
       18.56445614, 18.93599095, 19.31496138, 19.70151623, 20.09580729,
       20.4979894 , 20.90822046, 21.32666158, 21.75347706, 22.1888345 ,
       22.63290484, 23.08586248, 23.54788525, 24.0191546 , 24.49985557,
       24.99017693, 25.4903112 , 26.00045477, 26.52080797, 27.05157512,
       27.59296463, 28.1451891 , 28.70846537, 29.28301462, 29.86906245,
       30.466839  , 31.076579  , 31.69852186, 32.33291181, 32.97999795,
       33.64003438, 34.31328028, 35.        ])
N = 500
Lbox = 250
Nbox = 50
with_shotnoise= True
xHmax = 0.1

xvals = np.zeros(N)
yT21vals = np.zeros(N)
yPvals = np.zeros(N)
yzerocross = np.zeros(N)
for nid in range(N):
    outputs = import_random_model(nid, model='OIII',Lbox=Lbox,with_shotnoise=with_shotnoise,Nbox = Nbox, _R=None,include_partlion=True)

    xHv = outputs['xHv']
    p = outputs['p']
    p[np.isnan(p)] = 0.
    T21 = outputs['T21']
    T21[np.isnan(T21)] = 0.

    kvals = outputs['k_cross']
    rcr = outputs['r']

    r_int = RegularGridInterpolator((use_zvals, kvals[0]), rcr, bounds_error=False, fill_value=0.)

    T21max_index = list(T21).index(np.max(T21))
    zT21max = use_zvals[T21max_index]

    der_P = np.gradient(p,use_zvals)
    # plt.plot(use_zvals,der_P,'--')
    # plt.plot(use_zvals,p,'--')
    # plt.axhline(0)
    # plt.show()

    # Define saturation as max value
    sat_level = np.asarray(p[::-1]).max()

    drop_idx = 0 
    tol = 0.07 * sat_level  

    for i in range(len(p) - 1):
        if abs(p[i] - sat_level) > tol and abs(p[i+1] - sat_level) <= tol:
            drop_idx = i
    
    # first index where q drops below (sat_level - tol)
    zPdrop = use_zvals[drop_idx]
    zxH = interp1d(xHv, use_zvals, bounds_error=False, fill_value=0.)

    use_r = r_int((use_zvals,0.13))

    yzerocross[nid] = (crossing_brackets(use_zvals, use_r)[0][0]+crossing_brackets(use_zvals, use_r)[0][1])/2.

    xvals[nid] = zxH(1-xHmax)
    yT21vals[nid] = zT21max
    yPvals[nid] = zPdrop

save_path = '../run_random_parameters/random_maps_L250_N50/' 
save_parameters_path = save_path + 'par_values.dat' 
data = np.loadtxt(save_parameters_path, skiprows=1)
nid, epsstar, alphastar, betastar, fesc, LX = data.T

extra_info = []
for nid in range(N):
    extra_info.append(
        'eps = ' + str(epsstar[nid]) + ',\n'+\
        'alpha = ' + str(alphastar[nid]) + ',\n'+\
        'beta = ' + str(betastar[nid]) + ',\n'+\
        'fesc = ' + str(fesc[nid]) + ',\n'+\
        'LX = ' + str(LX[nid]) 
    )
colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

labels = [f"Point {i}" for i in range(N)]
colors_left = [colors[3]]*N
colors_right = [colors[-2]]*N

def create_figure(highlight_index=None):
        # Copy default colors
    colors_l = colors_left.copy()
    colors_r = colors_right.copy()

    # Highlight hovered point
    if highlight_index is not None:
        colors_l[highlight_index] = colors[5]
        colors_r[highlight_index] = colors[5]

    fig = make_subplots(rows=1, cols=3 )
    # Scatter left panel
    fig.add_trace(
        go.Scatter(
            x=xvals, y=yT21vals,
            mode="markers",
            name="z(max T21)",
            marker=dict(
                size=12,
                color=colors_l,    # <- use the array, not a single color
                line=dict(width=2, color="black")
            ),
            customdata=np.stack([labels, yPvals], axis=-1),
            hovertemplate=extra_info
        ),
        row=1, col=1
    )

    # Highlighted point (if any)
    if highlight_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[xvals[highlight_index]],
                y=[yT21vals[highlight_index]],
                mode="markers",
                marker=dict(size=16, color=colors[5], line=dict(width=3, color="black")),
                showlegend=False,
                hoverinfo="skip"  # optional: avoid duplicate hover tooltip
            ),
            row=1, col=1
        )

    # Main scatter (all points)
    fig.add_trace(
        go.Scatter(
            x=xvals, y=yPvals,
            mode="markers",
            marker=dict(size=12, color=colors_r, line=dict(width=2, color="black")),
            customdata=np.stack([labels, yPvals], axis=-1),
            hovertemplate=extra_info,
            name="z(P drop)"
        ),
        row=1, col=2
    )

    # Highlighted point (if any)
    if highlight_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[xvals[highlight_index]],
                y=[yPvals[highlight_index]],
                mode="markers",
                marker=dict(size=16, color=colors[5], line=dict(width=3, color="black")),
                showlegend=False,
                hoverinfo="skip"  # optional: avoid duplicate hover tooltip
            ),
            row=1, col=2
        )

    fig.add_trace(
        go.Scatter(
            x=xvals, y=yzerocross,
            mode="markers",
            marker=dict(size=12, color=colors_r, line=dict(width=2, color="black")),
            customdata=np.stack([labels, yzerocross], axis=-1),
            hovertemplate=extra_info,
            name="z(zero cross)"
        ),
        row=1, col=3
    )

    # Highlighted point (if any)
    if highlight_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[xvals[highlight_index]],
                y=[yzerocross[highlight_index]],
                mode="markers",
                marker=dict(size=16, color=colors[5], line=dict(width=3, color="black")),
                showlegend=False,
                hoverinfo="skip"  # optional: avoid duplicate hover tooltip
            ),
            row=1, col=3
        )


    fig.add_trace(
        go.Scatter(
            x=[5, 15],
            y=[5, 15],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False  # optional, hides from legend
        ),
        row=1, col=1
    )

    # Right panel diagonal
    fig.add_trace(
        go.Scatter(
            x=[5, 15],
            y=[5, 15],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    # Right panel diagonal
    fig.add_trace(
        go.Scatter(
            x=[5, 15],
            y=[5, 15],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        ),
        row=1, col=3
    )

    # Save interactive HTML
    fig.update_xaxes(title_text="z(xHII=10%)", row=1, col=1)
    fig.update_yaxes(title_text="z(max T21)", row=1, col=1)

    fig.update_xaxes(title_text="z(xHII=10%)", row=1, col=2)
    fig.update_yaxes(title_text="z(P drop)", row=1, col=2)

    # For left panel
    fig.update_xaxes(range=[6, 15], row=1, col=1)
    fig.update_yaxes(range=[6, 15], row=1, col=1)

    # For right panel
    fig.update_xaxes(range=[6, 15], row=1, col=2)
    fig.update_yaxes(range=[6, 15], row=1, col=2)
    fig.update_layout(
    width=1800,  # width in pixels
    height=900,  # height in pixels
        font=dict(
            family="Times New Roman, serif",
            size=16,
            color="black"
        )
    )

    fig.write_html("explore_parameters.html")
    return fig

# -------------------
# Initialize Dash
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    # Set initial figure so it shows immediately
    dcc.Graph(id="linked-scatter", figure=create_figure())
])

# Callback to update colors on hover
@app.callback(
    Output("linked-scatter", "figure"),
    Input("linked-scatter", "hoverData")
)
def update_colors(hoverData):
    highlight_index = None
    if hoverData:
        highlight_index = hoverData["points"][0]["pointIndex"]
    return create_figure(highlight_index)

app.run(debug=False, port=8051)


