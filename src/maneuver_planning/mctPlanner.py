import numpy as np

class State:
    def __init__(
            time,
            x_c,
            x_d,
            Sig_c,
            Sig_d,
            r_c, 
            r_d
            ):
        

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