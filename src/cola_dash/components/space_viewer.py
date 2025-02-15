from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import numpy as np
from PIL import Image
import plotly.graph_objects as go

import cola_dash.style as style
import utils.orbits as orbits

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
EARTH_IMAGE = Image.open("./src/assets/earth.jpeg")
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

        # Create layout
        self.content = html.Div([
            html.H2(
                children="Space Viewer",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Dropdown(
                    id="space-dropdown",
                    options=["Satellite", "Rocket Body"],
                    multi=True,
                ),
            ], style=style.DASH_1),               
            html.Div([
                dbc.Spinner([
                    dcc.Graph(
                        id="space-viewer",
                        figure=self.fig,
                        style={"height": "75vh"},
                    ),
                ]),
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
    
    def plot_orbit(self, satcat, x_ecef, y_ecef, z_ecef):
        return go.Scatter3d(
            x=x_ecef,
            y=y_ecef,
            z=z_ecef,
            mode="lines",
            marker=dict(size=2),
            line=dict(width=2),
            name=satcat
        )
    
    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("space-viewer", "figure"),
            Input("space-dropdown", "value")
        )
        def update_graph(value):
            """Create and update space viewer visualization."""

            # Clear all data except for Earth
            self.fig.data = self.fig.data[:1]

            # TODO: Query data based on satcat
            for i in range(1, 3):
                ecef = orbits.keplerian2ecef(*orbits.random_keplerian())
                self.fig.add_trace(self.plot_orbit(f"Satellite {i}", *ecef))
            return self.fig
