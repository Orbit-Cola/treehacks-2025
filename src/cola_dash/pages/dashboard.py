import dash
from dash import html, Output, Input, callback
import dash_bootstrap_components as dbc

from components.conjunction_screening import ConjunctionScreening
from components.space_viewer import SpaceViewer
import style

dash.register_page(__name__)

conjunction_screening = ConjunctionScreening()
space_viewer = SpaceViewer()

conjunction_screening.register_callbacks()
space_viewer.register_callbacks()

tabs = dbc.Tabs(
    id="tabs",
    active_tab="tab-1",
    children=[
        dbc.Tab(label="Space Viewer", tab_id="tab-1"),
        dbc.Tab(label="Conjunction Screening", tab_id="tab-2"),
    ],
    style=style.DASH_1,
)

@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_content(tab):
    """Render content for the selected tab."""
    if tab == "tab-1":
        return space_viewer.content
    elif tab == "tab-2":
        return conjunction_screening.content
    
layout = [
    tabs,
    dbc.Spinner(
        [
            html.Div(id="tab-content", className="p-4"),
        ],
        delay_show=100,
    ),
]
