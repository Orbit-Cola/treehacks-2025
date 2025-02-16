import json
import numpy as np
import os
import sys
from scipy.spatial import KDTree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import src.utils.database as db
from pc_calc import *
from tolerances import *

class RiskAnalyzer():
    """
    RiskAnalyzer: runs calcs to see what satellites are at risk of colliding
    Inputs:
    1) None!
    Outputs:
    1) concerns: Dictionary(satellite_id) = [[pair, Pc], ... ]
    """

    def __init__(self):
        self.risky_cases = []
        conn = db.create_conn()
        cursor = conn.cursor()
        self.get_jsons(cursor)
        self.collision_checker()
        self.amongus_json = []
        self.create_jsons()
        db.delete_conjunction_data(cursor=cursor)

        # Write all conjunctions to database
        print("Writing " + str(len(self.amongus_json)) + " Conjunction Event(s) to Database")
        db.upload_conjunction(cursor=cursor,conjunction_data=self.amongus_json)
        conn.commit()

    def get_jsons(self,cursor):
        # Pull propagator data from database and parse
        raw_propagator_data = db.get_propagation_data(cursor)
        self.sat_data = []

        # Read in data from database
        for obj in raw_propagator_data:
            json_data = json.loads(obj[3])
            self.sat_data.append(json_data)
        
        print("Read in " + str(len(self.sat_data)) + " TLE(s)")

    def process_jsons(self):
        """
        Operates over a single timestep
        Create 2 datastructures:
        1) sat_data: list of ints corresponding to satellite json info
        """

        # positions = []
        self.sat_data = []
        for data in self.jsons:
            with open(data, "r") as f:
                big_data = json.load(f)
                self.sat_data.append(big_data)

        # # convert positions to np.array
        # self.positions = np.vstack(positions)

    def collision_checker(self):
        """
        Conduct junction analysis across multiple every available time steps
        """
        num_times = len(self.sat_data[0]["time_utc"])
        
        # TODO: GET T
        for i in range(num_times):
        # for i in range(1):

            # Get position vector
            positions = []
            self.ind = i
            for sat in self.sat_data:
                positions.append(sat["position_eci_km"][i])

            positions = np.array(positions)
            risk_indices = self.find_close_pairs(positions)
            self.get_pcs(risk_indices, sat["time_utc"][i])

    def create_jsons(self):
        """Create jsons with our data"""
        for risky_case in self.risky_cases:
            file_name = str(risky_case["Satellite 1"]["Satellite catalog number"]) + "_" + str(risky_case["Satellite 2"]["Satellite catalog number"]) + "" + risky_case["time_utc"].replace(":", "").replace(" ", "_").replace("-","_")
            file_full_name = os.path.join("src/conjunction_analysis/conjunction_data/", file_name+".json")

            with open(file_full_name, "w") as outfile:
                json.dump(risky_case, outfile, indent=4)
            
            self.amongus_json.append((
                str(risky_case["Satellite 1"]["Satellite catalog number"]),
                str(risky_case["Satellite 2"]["Satellite catalog number"]),
                json.dumps(risky_case, indent=4)
            ))

    def find_close_pairs(self, positions):
        tree = KDTree(positions)
        pairs = tree.query_pairs(SAFE_DISTANCE)
        return list(pairs)
    
    def get_sat_object(self, sat_data):
        sat = satellite(id=sat_data["Satellite catalog number"],
                        name=sat_data["name"],
                        # size=sat_data["frontal_area_m2"]*1e-3,
                        size=sat_data["frontal_area_m2"],
                        pos=np.array(sat_data["position_eci_km"][self.ind]),
                        vel=np.array(sat_data["velocity_eci_km_s"][self.ind]),
                        cov_rtn=np.array(sat_data["covariance_position_rtn"][self.ind]))
        
        return sat
    
    def get_pcs(self, risk_indices, time):
        for pair in risk_indices:
            # TODO: Figure out how to pull data from database here
            sat1 = self.get_sat_object(sat_data=self.sat_data[pair[0]])
            sat2 = self.get_sat_object(sat_data=self.sat_data[pair[1]])
            prob_collision = calcPC(sat1=sat1,
                                    sat2=sat2).Pc
            if prob_collision >= SAFE_PC:
                # risky_case.append([(sat1.id, sat2.id), time])
                # risky_Pcs.append(prob_collision)
                risky_dict = {
                    "Satellite 1": {
                        "Satellite catalog number": sat1.id,
                        "Satellite name": sat1.name,
                        "position_eci_km": sat1.pos.tolist(),
                        "covariance_position_eci": sat1.cov.tolist(),
                        "covariance_rtn": sat1.cov_rtn.tolist(),
                        },
                    "Satellite 2": {
                        "Satellite catalog number": sat2.id,
                        "Satellite name": sat2.name,
                        "position_eci_km": sat2.pos.tolist(),
                        "position_rtn_km": (gcrs2ric(sat1.pos,sat1.vel)@(sat2.pos-sat1.pos)).tolist(),
                        "covariance_position_eci": sat2.cov.tolist(),
                        "covariance_rtn": sat2.cov_rtn.tolist(),
                        },
                    "time_utc": time,
                    "Pc_percentage": prob_collision*100
                    }
                self.risky_cases.append(risky_dict)
    
    def fake_jsons():
        """
        In leiu of actual covariances, this will generate some fabricated results
        """

    
risks = RiskAnalyzer()