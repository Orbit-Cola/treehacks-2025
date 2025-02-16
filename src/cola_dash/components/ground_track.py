from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src.cola_dash.components.db_helper import PROPAGATOR_DICT
import src.cola_dash.style as style

LAND_COLOR = "#007000"
WATER_COLOR = "#000070"
OPTIONS_DICT = {f"{satcat}: {json_data['name']}": satcat for satcat, json_data in PROPAGATOR_DICT.items()}
OPTIONS = list(OPTIONS_DICT.keys())

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
                    options=OPTIONS,
                    value=OPTIONS[0] if OPTIONS else None,
                    multi=True,
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
            Input("ground-dropdown", "value"),
        )
        def update_graph(value):
            """Update graph."""
            fig = go.Figure()
            if value:
                if type(value) is str:
                    value = [value,]
                for v in value:
                    obj = PROPAGATOR_DICT[OPTIONS_DICT[v]]
                    fig.add_trace(
                        go.Scattergeo(
                            lat=obj["latitude_deg"],
                            lon=obj["longitude_deg"],
                            text=v,
                            name="",
                            mode="lines",
                        )
                    )
            fig.update_layout(template="plotly_dark")
            fig.update_geos(
                lakecolor=WATER_COLOR,
                landcolor=LAND_COLOR,
                oceancolor=WATER_COLOR,
                showcountries=True,
                showocean=True,
            )
            fig.update_layout(
                margin=dict(l=10, r=10, b=10, t=10, pad=0, autoexpand=True),
            )
            return fig
