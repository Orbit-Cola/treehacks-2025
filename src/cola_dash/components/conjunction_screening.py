from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from src.cola_dash.components.db_helper import CONJUNCTION_LIST
OPTIONS_DICT = {
    f"Sat {CONJUNCTION_LIST[_i]['Satellite 1']['Satellite catalog number']}: {CONJUNCTION_LIST[_i]['Satellite 1']['Satellite name']} - "
    f"Sat {CONJUNCTION_LIST[_i]['Satellite 2']['Satellite catalog number']}: {CONJUNCTION_LIST[_i]['Satellite 2']['Satellite name']}": _i
    for _i in range(len(CONJUNCTION_LIST))
}
OPTIONS = list(OPTIONS_DICT.keys())

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
                    id="conjunction-selection",
                    options=OPTIONS,
                    value=OPTIONS[0] if OPTIONS else None,
                    multi=False,
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
                    bordered=True, 
                    striped=True, 
                    hover=True
                ),
            ], style=style.DASH_1),
        ])

        # Register callbacks on initialization
        self.register_callbacks()

    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("conjunction-geometry", "figure"),
            Input("conjunction-selection", "value")
        )
        def update_graph(value):
            """Update graph."""
            # Get conjunction dict
            conjunction_dict = CONJUNCTION_LIST[OPTIONS_DICT[value]]

            # Get covariance matrices
            cov1 = np.array(conjunction_dict["Satellite 1"]["covariance_rtn"])
            cov2 = np.array(conjunction_dict["Satellite 2"]["covariance_rtn"])

            # Get position of sat 2 in RIC frame for sat 1
            r_ric2 = np.array(conjunction_dict["Satellite 2"]["position_rtn_km"])

            # Create covariance ellipsoid for sat 1
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x1 = np.sqrt(cov1[0,0]) * np.outer(np.cos(u), np.sin(v))
            y1 = np.sqrt(cov1[1,1]) * np.outer(np.sin(u), np.sin(v))
            z1 = np.sqrt(cov1[2,2]) * np.outer(np.ones_like(u), np.cos(v))

            # Create covariance ellipsoid for sat 2
            x2 = np.sqrt(cov2[0,0]) * np.outer(np.cos(u), np.sin(v)) + r_ric2[0]
            y2 = np.sqrt(cov2[1,1]) * np.outer(np.sin(u), np.sin(v)) + r_ric2[1]
            z2 = np.sqrt(cov2[2,2]) * np.outer(np.ones_like(u), np.cos(v)) + r_ric2[2]

            primary = go.Surface(
                x=x1,
                y=y1,
                z=z1,
                opacity=0.5,
                colorscale=[[0, "green"], [1, "green"]],
                showlegend=True,
                showscale=False,
                name="Primary"
            )
            secondary = go.Surface(
                x=y2,
                y=z2,
                z=x2,
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

        @callback(
            Output("conjunction-table", "children"),
            Input("conjunction-selection", "value")  # If filtering by dropdown
        )
        def update_table(_):
            # Extract headers from the dictionary keys
            headers = [html.Th("Time (UTC)"),html.Th("Satellite 1"),html.Th("Satellite 2"),html.Th("Pc (%)")]
            thead = html.Thead(html.Tr(headers))
            
            # Check if Conjunction values exist
            if not CONJUNCTION_LIST:
                return dbc.Table([thead], bordered=True, striped=True, hover=True)

            # Create rows for the table body
            rows = [
                html.Tr([
                    html.Td(str(CONJUNCTION_LIST[_i]["time_utc"])),
                    html.Td(str(CONJUNCTION_LIST[_i]["Satellite 1"]["Satellite catalog number"]) + ": " + str(CONJUNCTION_LIST[_i]["Satellite 1"]["Satellite name"])),
                    html.Td(str(CONJUNCTION_LIST[_i]["Satellite 2"]["Satellite catalog number"]) + ": " + str(CONJUNCTION_LIST[_i]["Satellite 2"]["Satellite name"])),
                    html.Td(str(CONJUNCTION_LIST[_i]["Pc_percentage"]))
                ])
                for _i in range(len(CONJUNCTION_LIST))
            ]
            tbody = html.Tbody(rows)

            # Return the full table component
            return dbc.Table([thead, tbody], bordered=True, striped=True, hover=True)