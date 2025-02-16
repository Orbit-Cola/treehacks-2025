from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

import src.cola_dash.style as style

class ConjunctionScreening:
    """Conjunction Screening page view."""

    def __init__(self):
        self.content = html.Div([
            html.H2(
                children="Conjunction Screening",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Dropdown(
                    id="dropdown-selection",
                ),
            ], style=style.DASH_1),
            html.Div([
                dbc.Spinner([
                    dcc.Graph(
                        id="conjunction-geometry",
                        style={"height": "75vh"},
                    ),
                ]),
            ], style=style.DASH_1),
            html.Div([
                dbc.Table(
                    id="conjunction-table",
                    ),
            ], style=style.DASH_1),
        ])

        # Register callbacks on initialization
        self.register_callbacks()

    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("conjunction-geometry", "figure"),
            Input("dropdown-selection", "value")
        )
        def update_graph(value):
            """Update graph."""
            # TODO: Make real covariance ellipsoids
            a = 2  # Semi-axis along x
            b = 3  # Semi-axis along y
            c = 4  # Semi-axis along z

            # Create the grid of points
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = a * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = c * np.outer(np.ones_like(u), np.cos(v))

            primary = go.Surface(
                x=x,
                y=y,
                z=z,
                opacity=0.5,
                colorscale=[[0, "green"], [1, "green"]],
                showlegend=True,
                showscale=False,
                name="Primary"
            )
            secondary = go.Surface(
                x=y + b,
                y=z + c,
                z=x + a,
                opacity=0.25,
                colorscale=[[0, "red"], [1, "red"]],
                showlegend=True,
                showscale=False,
                name="Secondary"
            )

            fig = go.Figure(data=[primary, secondary])
            fig.update_layout(
                template="plotly_dark",
                title='Conjunction Geometry',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                scene=dict(
                    xaxis_title='Radial [km]',
                    yaxis_title='In-Track [km]',
                    zaxis_title='Cross-Track [km]',
                    aspectmode='auto'
                )
            )
            return fig
