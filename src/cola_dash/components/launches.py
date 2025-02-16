from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from src.cola_dash.components.db_helper import LAUNCHES_DATA
import src.cola_dash.style as style

class Launches:
    """Launches page view."""

    def __init__(self):
        self.df = pd.DataFrame(LAUNCHES_DATA, columns =['Launch', 'NET', "Window Close", 'Window Open', "Launch Provider", "Rocket"])
        self.content = html.Div([
            html.H2(
                children="Recent and Upcoming Launches",
                style=style.HEADING_2,
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        dcc.Graph(
                            figure= px.histogram(self.df["Launch Provider"], x="Launch Provider", template="plotly_dark")
                        ),
                    ),
                    dbc.Col(
                        dcc.Graph(
                            figure= px.histogram(self.df["Rocket"], x="Rocket", template="plotly_dark")
                        ),
                    ),
                ]
            ),
            html.Hr(),
            dbc.Table.from_dataframe(self.df, hover=True, className="table-dark")
        ])
