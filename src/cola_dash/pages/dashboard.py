import dash
from dash import html, Output, Input, callback
import dash_bootstrap_components as dbc

from src.cola_dash.components.conjunction_screening import ConjunctionScreening
from src.cola_dash.components.ground_track import GroundTrack
from src.cola_dash.components.launches import Launches
from src.cola_dash.components.space_viewer import SpaceViewer
import src.cola_dash.style as style

dash.register_page(__name__)

conjunction_screening = ConjunctionScreening()
ground_track = GroundTrack()
launches = Launches()
space_viewer = SpaceViewer()

tabs = dbc.Tabs(
    id="tabs",
    active_tab="tab-1",
    children=[
        dbc.Tab(label="Space Viewer", tab_id="tab-1"),
        dbc.Tab(label="Conjunction Screening", tab_id="tab-2"),
        dbc.Tab(label="Ground Track", tab_id="tab-3"),
        dbc.Tab(label="Launches", tab_id="tab-4"),
    ],
    style=style.DASH_1,
    persistence=True,
    persistence_type='session',
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
    elif tab == "tab-3":
        return ground_track.content
    elif tab == "tab-4":
        return launches.content
    
layout = [
    tabs,
    html.Hr(),
    html.Div(id="tab-content", className="p-4"),
]
