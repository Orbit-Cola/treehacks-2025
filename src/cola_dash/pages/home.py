import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/", title="Orbit Cola")

layout = dbc.Container([
    dbc.Row([
        html.H1(
            children="Keeping space open to all 🚀",
            style={"text-align": "center"}
        ),
        html.Div([
            dbc.Button("Launch Dashboard", href="/dashboard", size="lg"),
        ], style={"display": "flex", "justify-content": "center"}),
    ], justify="center", align="center", className="h-50"),
], style={"height": "100vh"})
