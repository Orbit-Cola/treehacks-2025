import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

import cola_dash.style as style

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.VAPOR],
    use_pages=True,
    pages_folder="cola_dash/pages",
    suppress_callback_exceptions=True
)

header = html.A(
    href="/",
    style=style.CLEAN_HREF,
    children=[
        html.H1(children="Orbit Cola 🥤", style=style.HEADING_1)
    ]
)

footer = html.Footer(
    children="Thanks for checking out Orbit Cola!",
    style=style.FOOTER,
)

app.layout = html.Div([
    header,
    dash.page_container,
    footer,
])

if __name__ == "__main__":
    app.run(debug=True)
