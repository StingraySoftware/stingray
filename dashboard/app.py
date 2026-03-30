from dash import Dash, html, dcc, Output, Input, callback
import plotly.express as px
import pandas as pd
from stingray import Lightcurve, Crossspectrum
import numpy as np

### Had to import dash and pandas separately (make a new category in setup.cfg)

dt = 0.01
time = np.arange(0, 100, dt)

freq = 0.2
amplitude = 100.0

counts = amplitude * np.sin(2 * np.pi * freq * time)
lc = Lightcurve(time, counts, dt=dt)

df_lc = pd.DataFrame({"time": lc.time, "counts": lc.counts})

cs = Crossspectrum(lc, lc)
df_cs_mag = pd.DataFrame({"frequency": cs.freq, "magnitude": np.abs(cs.power)})
df_cs_phase = pd.DataFrame({"frequency": cs.freq, "phase": np.angle(cs.power)})

app = Dash(__name__)

graph_styles = {"width": "50vw", "height": "50vh"}

app.layout = html.Div(
    [
        html.H1("Stingray Data", style={"textAlign": "center"}),
        html.Div(
            className="wrapper",
            children=[
                dcc.Graph(
                    id="graph_lc", figure=px.line(df_lc, x="time", y="counts"), style=graph_styles
                ),
                dcc.Graph(
                    id="graph_cs_mag",
                    figure=px.line(df_cs_mag, x="frequency", y="magnitude"),
                    style=graph_styles,
                ),
                dcc.Graph(
                    id="graph_cs_phase",
                    figure=px.line(df_cs_phase, x="frequency", y="phase"),
                    style=graph_styles,
                ),
            ],
            style={"display": "flex", "flex-direction": "column", "alignItems": "center"},
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
