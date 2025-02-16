import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/", title="Orbit Cola")

# TODO: CSS nightmare
layout = dbc.Container([
    dbc.Row([
        html.H1(
            children="Keeping space open to all 🚀",
            style={"text-align": "center"}
        ),
        html.Img(src="assets/logo.png", style={"height": "25vh", "width": "auto", "display": "block"}),
        html.Div([
            dbc.Button("Launch Dashboard", href="/dashboard", size="lg"),
        ], style={"display": "flex", "justify-content": "center"}),
    ], justify="center", align="center", className="h-50"),
], style={"height": "100vh"})
