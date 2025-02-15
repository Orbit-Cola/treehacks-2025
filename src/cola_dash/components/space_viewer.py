from dash import dcc, html, Output, Input, callback
import pandas as pd
import plotly.express as px

import style

# LATEST_PATH = "https://orbitcola.com/latest.csv"
LATEST_PATH = "./res/latest.csv"

class SpaceViewer:
    """Space Viewer page view with 3D orbit visualization and space safety statistics."""

    def __init__(self):
        self.latest_df = pd.read_csv(LATEST_PATH)
        self.min = self.latest_df.time.min()
        self.max = self.latest_df.time.max()
        self.step = (self.max - self.min) / (len(self.latest_df.time.unique()) - 1)

        self.content = html.Div([
            html.H1(
                children="Space Viewer",
                style=style.HEADING_2,
            ),
            html.Div([
                dcc.Slider(
                    min=self.min,
                    max=self.max,
                    step=self.step,
                    value=self.min,
                    id="slider-selection",
                    updatemode="drag",
                ),
                dcc.Graph(
                    id="space-viewer",
                ),
            ], style=style.DASH_1),
        ])

    def register_callbacks(self):
        """Register the callbacks for this class."""
        @callback(
            Output("space-viewer", "figure"),
            Input("slider-selection", "value")
        )
        def update_graph(value):
            """Create and update space viewer visualization."""
            dff = self.latest_df[self.latest_df.time==value]
            fig = px.scatter_3d(
                data_frame=dff,
                x=dff.x,
                y=dff.y,
                z=dff.z,
                color=dff.id,
                hover_name="id",
                range_x=(-10, 10),
                range_y=(-10, 10),
                range_z=(-10, 10),
                template="plotly_dark",
            )
            fig.update_layout(
                scene_aspectmode="manual",
                scene_aspectratio=dict(x=1, y=1, z=1),
            )
            return fig
