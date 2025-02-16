import json
import numpy as np
import os
from scipy.spatial import KDTree

from tolerances import *
from pc_calc import *

class RiskAnalyzer():
    """
    RiskAnalyzer: runs calcs to see what satellites are at risk of colliding
    Inputs:
    1) None!
    Outputs:
    1) concerns: Dictionary(satellite_id) = [[pair, Pc], ... ]
    """

    def __init__(self):
        self.get_jsons()
        self.process_jsons()
        risk_indices = self.find_close_pairs()
        self.concerns, self.Pcs = self.get_pcs(risk_indices)

    def get_jsons(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(script_dir)
        json_dir = os.path.join(src_dir, "sgp4", "sqp4_json")
        self.jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    def process_jsons(self):
        """
        Create 2 datastructures:
        1) positions: list of np.array(3) corresponding to positions of different sats
        2) sat_data: list of ints corresponding to satellite json info
        """

        positions = []
        self.sat_data = []
        for data in self.jsons:
            with open(data, "r") as f:
                big_data = json.load(f)

                positions.append(np.array(big_data[0]["position_eci_km"]))
                self.sat_data.append(big_data[0])

        # convert positions to np.array
        self.positions = np.vstack(positions)

    def find_close_pairs(self):
        tree = KDTree(self.positions)
        pairs = tree.query_pairs(SAFE_DISTANCE)
        return list(pairs)
    
    def get_sat_object(self, sat_data):
        sat = satellite(id=sat_data["Satellite catalog number"],
                        size=sat_data["frontal_area_m2"]*1e-6,
                        pos=np.array(sat_data["position_eci_km"][0]),
                        vel=np.array(sat_data["velocity_eci_km_s"][0]),
                        cov=np.array(sat_data["covariance_position_rtn"][0]))
        
        return sat

    
    def get_pcs(self, risk_indices):
        risky_pairs = []
        risky_Pcs = []
        for pair in risk_indices:
            # TODO: Figure out how to pull data from database here
            sat1 = self.get_sat_object(sat_data=self.sat_data[pair[0]])
            sat2 = self.get_sat_object(sat_data=self.sat_data[pair[1]])
            prob_collision = calcPC(sat1=sat1,
                                    sat2=sat2).Pc
            if prob_collision >= SAFE_PC:
                risky_pairs.append(pair)
                risky_Pcs.append(prob_collision)
        return risky_pairs, risky_Pcs
    
risks = RiskAnalyzer()
print(risks.Pcs, risks.concerns)

# Stress test
# for n in range(10000):
#     sat1 = satellite("A")
#     sat1.pos = np.random.rand(3) * 10
#     sat1.vel = np.random.rand(3) * 10
#     sat1.size = np.random.rand(1) * 1
#     sat1.cov = np.random.rand(3,3)
#     sat2 = satellite("B")
#     sat2.pos = sat1.pos + np.random.rand(3)
#     sat2.vel = sat1.pos + np.random.rand(3)
#     sat2.size = np.random.rand(1) * 1
#     sat2.cov = np.random.rand(3,3)

#     pc_calcs = calcPC(sat1=sat1, sat2=sat2)

#     print(pc_calcs.Pc)