from dash import dcc, html, Output, Input, Patch, callback
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from src.cola_dash.components.db_helper import PROPAGATOR_DICT, PROPAGATOR_TIMESTEPS
import src.cola_dash.style as style

# From https://community.plotly.com/t/applying-full-color-image-texture-to-create-an-interactive-earth-globe/60166
EARTH_COLORSCALE = [
    [0.0, "rgb(30, 59, 117)"],
    [0.1, "rgb(46, 68, 21)"],
    [0.2, "rgb(74, 96, 28)"],
    [0.3, "rgb(115,141,90)"],
    [0.4, "rgb(122, 126, 75)"],
    [0.6, "rgb(122, 126, 75)"],
    [0.7, "rgb(141,115,96)"],
    [0.8, "rgb(223, 197, 170)"],
    [0.9, "rgb(237,214,183)"],
    [1.0, "rgb(255, 255, 255)"],
]
EARTH_IMAGE = Image.open("./src/cola_dash/assets/earth.jpeg")
EARTH_TEXTURE = np.asarray(EARTH_IMAGE.resize((np.array(EARTH_IMAGE.size) / 4).astype(int), Image.LANCZOS)).T

class SpaceViewer:
    """Space Viewer page view with 3D orbit visualization and space safety statistics."""

    def __init__(self):

        # Initialize 3D plot with Earth
        self.earth = self.plot_earth()
        self.fig = go.Figure(data=[self.earth])
        self.fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, visible=False),
                yaxis=dict(showgrid=False, showticklabels=False, visible=False),
                zaxis=dict(showgrid=False, showticklabels=False, visible=False),
            ),
        )
        self.init_idx = 0
        for obj in PROPAGATOR_DICT.values():
            name = obj["name"]
            r_eci = np.array(obj["position_eci_km"])
            self.fig.add_trace(self.plot_object(name, r_eci[:, 0], r_eci[:, 1], r_eci[:, 2], self.init_idx))
        # Create layout
        self.content = html.Div([
            html.H2(
                children="Space Viewer",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Slider(
                    min=0,
                    max=len(PROPAGATOR_TIMESTEPS) - 1,
                    step=1,
                    value=self.init_idx,
                    id="space-slider",
                    marks={
                        i: {"label": t, "style": {"font-size": 10}}
                        for i, t in enumerate(PROPAGATOR_TIMESTEPS)
                        if t[-5:-3] == "00" or t[-5:-3] == "30" or (t[-4] == "0" and len(PROPAGATOR_TIMESTEPS) <= 60)
                    },
                    updatemode="drag",
                    persistence=True,
                    persistence_type='session',
                )
            ], style=style.DASH_1),               
            html.Div([
                dcc.Graph(
                    id="space-viewer",
                    figure=self.fig,
                    style={"height": "75vh"},
                ),
            ], style=style.DASH_1),
        ])

        # Register callbacks on initialization
        self.register_callbacks()

    def plot_earth(self): 
        N_lat = int(EARTH_TEXTURE.shape[0])
        N_lon = int(EARTH_TEXTURE.shape[1])
        theta = np.linspace(0, 2 * np.pi, N_lat)
        phi = np.linspace(0, np.pi, N_lon)
        x = 6378 * np.outer(np.cos(theta), np.sin(phi))
        y = 6378 * np.outer(np.sin(theta), np.sin(phi))
        z = 6357 * np.outer(np.ones(N_lat), np.cos(phi))
        return go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=EARTH_TEXTURE,
            colorscale=EARTH_COLORSCALE,
            showscale=False,
            hoverinfo="none",
        )
    
    def plot_orbit(self, name, x_eci, y_eci, z_eci, color):
        return go.Scatter3d(
            x=x_eci,
            y=y_eci,
            z=z_eci,
            mode="lines",
            marker=dict(size=2),
            line=dict(width=2),
            name=name
        )

    def plot_object(self, name, x_eci, y_eci, z_eci, step):
        return go.Scatter3d(
            x=x_eci[step:step + 1],
            y=y_eci[step:step + 1],
            z=z_eci[step:step + 1],
            mode="markers",
            marker=dict(size=2),
            name=name
        )
    
    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("space-viewer", "figure"),
            Input("space-slider", "value"),
        )
        def update_graph(value):
            """Create and update space viewer visualization."""

            p = Patch()
            for idx, obj in enumerate(PROPAGATOR_DICT.values()):
                r_eci = np.array(obj["position_eci_km"])
                p["data"][idx + 1]['x'] = r_eci[value - 1, 0:1]
                p["data"][idx + 1]['y'] = r_eci[value - 1, 1:2]
                p["data"][idx + 1]['z'] = r_eci[value - 1, 2:3]
            return p
