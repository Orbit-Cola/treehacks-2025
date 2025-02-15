from scipy.spatial import KDTree
import numpy as np
from tolerances import *
from pc_calc import *

class RiskAnalyzer():

    def __init__(self, positions, database):
        self.positions = positions
        self.database = database
        risk_indices = self.find_close_pairs()
        self.concerns, self.Pcs = self.get_pcs(risk_indices)


    @staticmethod
    def find_close_pairs(self):
        tree = KDTree(self.positions)
        pairs = tree.query_pairs(SAFE_DISTANCE)
        return list(pairs)
    
    @staticmethod
    def get_pcs(self, risk_indices):
        risky_pairs = []
        risky_Pcs = []
        for pair in risk_indices:
            # TODO: Figure out how to pull data from database here
            sat1 = self.database[pair[0]]
            sat2 = self.database[pair[1]]
            prob_collision = calcPC(sat1=sat1,
                                    sat2=sat2).Pc
            if prob_collision.Pc <= SAFE_PC:
                risky_pairs.append(pair)
                risky_Pcs.append(prob_collision)
        return risky_pairs, risky_Pcs