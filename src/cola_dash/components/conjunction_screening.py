from dash import dcc, html, Output, Input, callback
import pandas as pd
import plotly.express as px

import style

CSV_PATH = "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv"

class ConjunctionScreening:
    """Conjunction Screening page view."""

    def __init__(self):
        self.df = pd.read_csv(CSV_PATH)
        self.content = html.Div([
            html.H1(
                children="Conjunction Screening",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Dropdown(
                    self.df.country.unique(),
                    "Canada",
                    id="dropdown-selection",
                ),
            ], style=style.DASH_1),
            html.Div([
                dcc.Graph(
                    id="conjunction-screening",
                ),
            ], style=style.DASH_1),
        ])

    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("conjunction-screening", "figure"),
            Input("dropdown-selection", "value")
        )
        def update_graph(value):
            """Update graph."""
            dff = self.df[self.df.country==value]
            return px.line(dff, x="year", y="pop", template="plotly_dark")
