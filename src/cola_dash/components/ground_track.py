from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import plotly.express as px

import cola_dash.style as style
import utils.orbits as orbits

LAND_COLOR = "#007000"
WATER_COLOR = "#000070"

class GroundTrack:
    """Ground Track page view."""

    def __init__(self):
        self.content = html.Div([
            html.H2(
                children="Ground Track",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Dropdown(
                    id="ground-dropdown",
                    options=["Satellite", "Rocket Body"],
                ),
            ], style=style.DASH_1),
            html.Div([
                dbc.Spinner([
                    dcc.Graph(
                        id="ground-track",
                        style={"height": "75vh"},
                    ),
                ]),
            ], style=style.DASH_1),
        ])

        # Register callbacks on initialization
        self.register_callbacks()

    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("ground-track", "figure"),
            Input("ground-dropdown", "value")
        )
        def update_graph(value):
            """Update graph."""
            lat, lon, _ = orbits.keplerian2latlon(*orbits.random_keplerian())
            fig = px.line_geo(lat=lat, lon=lon, template="plotly_dark")
            fig.update_geos(
                lakecolor=WATER_COLOR,
                landcolor=LAND_COLOR,
                oceancolor=WATER_COLOR,
                showcountries=True,
                showocean=True,
            )
            fig.update_traces(line_color="red")
            fig.update_layout(
                margin=dict(l=10, r=10, b=10, t=10, pad=0, autoexpand=True),
            )
            return fig
