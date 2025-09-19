from plots_paper import * 
from generate_maps import import_random_model, alphastar_fid, epsstar_fid, betastar_fid, fesc_fid, LX_fid

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output



use_zvals = zvals
N = 100
Lbox = 750
Nbox = 150
with_shotnoise= True

xvals = np.zeros(N)
yT21vals = np.zeros(N)
yPvals = np.zeros(N)
for nid in range(N):
    outputs = import_random_model(nid, model='OIII',Lbox=Lbox,with_shotnoise=with_shotnoise,Nbox = Nbox, _R=None,include_partlion=True)

    xHv = outputs['xHv']
    p = outputs['p']
    p[np.isnan(p)] = 0.
    T21 = outputs['T21']
    T21[np.isnan(T21)] = 0.

    T21max_index = list(T21).index(np.max(T21))
    zT21max = use_zvals[T21max_index]

    # Define saturation as max value
    sat_level = np.asarray(p[::-1]).max()

    drop_idx = 0 
    tol = 0.04 * sat_level  # 2% tolerance, adjust as needed

    for i in range(len(p) - 1):
        # current value NOT close to sat_level
        if abs(p[i] - sat_level) > tol and abs(p[i+1] - sat_level) <= tol:
            drop_idx = i
    
    # first index where q drops below (sat_level - tol)
    zPdrop = use_zvals[drop_idx]
    zxH = interp1d(xHv, use_zvals, bounds_error=False, fill_value=0.)

    xvals[nid] = zxH(1-xHmax)
    yT21vals[nid] = zT21max
    yPvals[nid] = zPdrop

save_path = './run_random_parameters/' +  'random_maps_L' + str(Lbox) +'_N' + str(Nbox) +  '/'
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

    fig = make_subplots(rows=1, cols=2, )
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
            customdata=np.stack([labels, yT21vals], axis=-1),
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

    return fig

# -------------------
# Initialize Dash
app = dash.Dash(__name__)

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


