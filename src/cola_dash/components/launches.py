from dash import html

class Launches:
    """Launches page view."""

    def __init__(self):
        self.content = html.Div([])

        # Register callbacks on initialization
        self.register_callbacks()

    def register_callbacks(self):
        """Register the callbacks for this class."""
        pass
