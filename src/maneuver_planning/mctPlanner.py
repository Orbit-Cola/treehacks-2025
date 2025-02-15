import numpy as np


class mctPlanner:
    def __init__(
        self,
        state_initial,  # initial state
        action_ranges,
    ):
        """
        We'll start somewhere, and then we'll use the policy to determine
        the next action to take
        """
        self.action_size = len(action_ranges)  

        # Action bounds
        self.action_ranges = action_ranges