from dash import dcc, html, Output, Input, callback, exceptions
import dash_bootstrap_components as dbc
import numpy as np
from PIL import Image
import plotly.graph_objects as go

import style

EARTH_COLORSCALE = [
    [0.0, 'rgb(30, 59, 117)'],
    [0.1, 'rgb(46, 68, 21)'],
    [0.2, 'rgb(74, 96, 28)'],
    [0.3, 'rgb(115,141,90)'],
    [0.4, 'rgb(122, 126, 75)'],
    [0.6, 'rgb(122, 126, 75)'],
    [0.7, 'rgb(141,115,96)'],
    [0.8, 'rgb(223, 197, 170)'],
    [0.9, 'rgb(237,214,183)'],
    [1.0, 'rgb(255, 255, 255)'],
]
EARTH_TEXTURE = np.asarray(Image.open('./src/cola_dash/assets/earth.jpeg')).T

class SpaceViewer:
    """Space Viewer page view with 3D orbit visualization and space safety statistics."""

    def __init__(self):
        self.globe = self.earth()
        self.fig = go.Figure(data=[self.globe])
        self.fig.update_layout(
            template="plotly_dark",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, visible=False),
                yaxis=dict(showgrid=False, showticklabels=False, visible=False),
                zaxis=dict(showgrid=False, showticklabels=False, visible=False),
            ),
        )
        self.content = html.Div([
            html.H1(
                children="Space Viewer",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Dropdown(
                    options=["Satellite", "Rocket Body"],
                    id="space-dropdown",
                ),
                dbc.Spinner(
                    [
                        html.Div([
                            dcc.Graph(
                                id="space-viewer",
                                figure=self.fig,
                            ),
                        ], style=style.DASH_1),
                    ],
                ),
            ]),
        ])

    def earth(self): 
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
    
    def get_keplerian(self, satcat):
        # TODO: Query data based on satcat
        a = np.random.randint(7200, 10000)
        e = np.random.rand() / 4
        inc = np.radians(np.random.randint(0, 90))
        raan = np.radians(np.random.randint(0, 90))
        om = np.radians(np.random.randint(0, 90))
        return a, e, inc, raan, om
    
    def orbit_keplerian(self, satcat, a, e, inc, raan, om, n=100):
        true_anomaly = np.linspace(0, 2 * np.pi, n)
        r = a * (1 - e ** 2) / (1 + e * np.cos(true_anomaly))
        x_eci = r * (np.cos(om + true_anomaly))  # Not sure if actually ECI
        y_eci = r * (np.sin(om + true_anomaly))
        z_eci = np.zeros(n)
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(inc), -np.sin(inc)],
                    [0, np.sin(inc), np.cos(inc)]])
        Rz_raan = np.array([[np.cos(raan), -np.sin(raan), 0],
                        [np.sin(raan), np.cos(raan), 0],
                        [0, 0, 1]])
        coords_ecef = []
        for i in range(n):
            pos_eci = np.array([x_eci[i], y_eci[i], z_eci[i]])
            pos_ecef = np.dot(Rz_raan, np.dot(Rx, pos_eci))
            coords_ecef.append(pos_ecef)
        coords_ecef = np.array(coords_ecef)
        x_ecef = coords_ecef[:, 0]
        y_ecef = coords_ecef[:, 1]
        z_ecef = coords_ecef[:, 2]
        return self.orbit(satcat, x_ecef, y_ecef, z_ecef)
    
    def orbit(self, satcat, x_ecef, y_ecef, z_ecef):
        return go.Scatter3d(
            x=x_ecef,
            y=y_ecef,
            z=z_ecef,
            mode='lines',
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
            if value is None:
                raise exceptions.PreventUpdate  # TODO: Implement filtering
