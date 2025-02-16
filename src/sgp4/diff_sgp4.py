import sys
import os

# Get the absolute path two levels up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from astropy.coordinates import TEME, ITRS, GCRS, CartesianDifferential, CartesianRepresentation, EarthLocation
import astropy.units as u
import numpy as np
from datetime import datetime, timedelta
import json
import dsgp4
from dsgp4.util import days2mdhms
import torch
torch.set_default_dtype(torch.float32)
import time
import src.utils.database as database

class BulkTLE:
    """
    Class to bulk propagate TLE data
    """
    def __init__(self,tle_str_list:list):
        """
        BulkTLE class constructor function

        Parameters
        ----------
        tle_txt: str
            The address of the text file containing bulk TLEs
        """
        # Print start
        print("\n--------------------------------------------------")
        print("Starting Bulk Propagation Operations")
        print("--------------------------------------------------")

        # Global tle list
        self.tle_list = []

        # Open text file and record all TLEs
        tle_list_init = [None]*len(tle_str_list)
        for _i in range(len(tle_str_list)):
            tle_list_init[_i]=dsgp4.tle.TLE(tle_str_list[_i])

        # Get rid of deepspace TLEs
        for _tle in tle_list_init:
            # Calculate auxiliary epoch quantities
            eccsq = _tle._ecco**2
            omeosq = 1 - eccsq
            rteosq = np.sqrt(omeosq)
            cosio = np.cos(_tle._inclo)
            cosio2 = cosio**2

            # Get WGS-84 constants
            tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2 = dsgp4.util.get_gravity_constants("wgs-84")
            x2o3 = 2./3.

            # Calc no_unkozai parameter
            ak = (xke / _tle._no_kozai) ** x2o3
            d1 = (
                0.75 * j2 * (3 * cosio2 - 1) / (rteosq * omeosq)
            )
            delta = d1 / (ak**2)
            adel = ak * (1 - delta**2 - delta * (1 / 3 + 134 * delta**2 / 81))
            delta = d1 / (adel**2)
            no_unkozai = _tle._no_kozai / (1 + delta)

            # Check deep space condition
            deep_space_condition = (2 * np.pi / no_unkozai >= 225.0)
            if not deep_space_condition:
                self.tle_list.append(_tle)
        print("\nPruned Deep Space Orbits: " + str(len(tle_list_init)-len(self.tle_list)))

        # Extract all epochs
        self.tle_epochs = []
        for _tle in self.tle_list:
            month, day, hour, minute, second = days2mdhms(_tle.epoch_year,_tle.epoch_days)
            microseconds = int((second - int(second)) * 1e6)
            self.tle_epochs.append(datetime(
                year=_tle.epoch_year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=int(second),
                microsecond=microseconds
            ))
        
        # Report number of TLEs
        print("\nNumber of TLEs: " + str(len(self.tle_list)))

        # json tuple
        self.json_tuple = []
        self.cd = 0.47/100 # Sphere
        self.m = 50. # [kg]

    def rotation_matrix(self,state):
        """
        Computes the UVW rotation matrix.

        Args:
            state (`numpy.array`): numpy array of 2 rows and 3 columns, where
                                        the first row represents position, and the second velocity.

        Returns:
            `numpy.array`: numpy array of the rotation matrix from the cartesian state.
        """
        r, v = state[0], state[1]
        u = r / np.linalg.norm(r)
        w = np.cross(r, v)
        w = w / np.linalg.norm(w)
        v = np.cross(w, u)
        return np.vstack((u, v, w))

    def from_cartesian_to_rtn(self,state, cartesian_to_rtn_rotation_matrix=None):
        """
        Converts a cartesian state to the RTN frame.

        Args:
            state (`numpy.array`): numpy array of 2 rows and 3 columns, where the first row represents position, and the second velocity.
            cartesian_to_rtn_rotation_matrix (`numpy.array`): numpy array of the rotation matrix from the cartesian state. If None, it is computed.

        Returns:
            `numpy.array`: numpy array of the RTN state.
        """
        # Use the supplied rotation matrix if available, otherwise compute it
        if cartesian_to_rtn_rotation_matrix is None:
            cartesian_to_rtn_rotation_matrix = self.rotation_matrix(state)
        r, v = state[0], state[1]
        r_rtn = np.dot(cartesian_to_rtn_rotation_matrix, r)
        v_rtn = np.dot(cartesian_to_rtn_rotation_matrix, v)
        return np.stack([r_rtn, v_rtn]), cartesian_to_rtn_rotation_matrix
    
    def bulk_propagate(self,times:list,json_dir:str):
        """
        Bulk propagate the self.tle_list of TLEs over the list of input times

        Parameters
        ----------
        times: list
            List of desired datetimes as datetime
        json_dir: str
            The file to store propagated data as json
        """
        # Make json directory
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # Prepare a local list for bulk propagation
        tle_batch_list = []
        for _tle in self.tle_list:
            tle_batch_list+=[_tle]*len(times)
        
        # Get timestamps in terms of time since epoch
        tsinces = []
        for tle_epoch in self.tle_epochs:
            tsince_list = [(dt - tle_epoch).total_seconds() / 60. for dt in times]
            tsinces.append(tsince_list)

        # Convert to a tensor for batch processing
        tsinces_tensor = torch.tensor(tsinces).flatten()
        
        # Initialize TLE batch
        tle_elements,tle_batch=dsgp4.initialize_tle(tle_batch_list,with_grad=True)

        # Propagate batch
        print("\nDesired Propagation Timestamps: " + str(len(times)))
        print("\nRunning TLE Batch Propagation...")
        prop_time_start = time.time()
        states_teme = dsgp4.propagate_batch(tle_batch,tsinces_tensor)
        prop_time_end = time.time()
        print("Finished in " + str(prop_time_end-prop_time_start) + " secs")

        # Propagate covariance matrices
        print("\nRunning TLE Covariance Propagation...")
        cov_time_start = time.time()
        Cov_xyz=np.zeros((len(tle_batch_list),6,6))
        Cov_rtn_pos=np.zeros((len(tle_batch_list),3,3))

        # Initial TLE covariance matrix:
        Cov_tle=np.identity(9) * 1/np.sqrt(3) # [km]
        frob_norms_pos=np.zeros((len(tle_batch_list),))

        dx_dtle = torch.zeros((len(tle_batch_list),6,9))
        for k in range(len(tle_batch_list)):
            for i in range(6):
                tle_elements[k].grad=None
                states_teme[k].flatten()[i].backward(retain_graph=True)
                dx_dtle[k,i,:] = torch.clamp(tle_elements[k].grad, min=-1.5, max=1.5) # [km/rad]
        
        for idx, tle in enumerate(tle_batch_list):
            state_rtn, cartesian_to_rtn_rotation_matrix = self.from_cartesian_to_rtn(states_teme[idx].detach().numpy())
            # I construct the 6x6 rotation matrix from cartesian -> RTN
            transformation_matrix_cartesian_to_rtn = np.zeros((6,6))
            transformation_matrix_cartesian_to_rtn[0:3, 0:3] = cartesian_to_rtn_rotation_matrix
            transformation_matrix_cartesian_to_rtn[3:,3:] = cartesian_to_rtn_rotation_matrix

            Cov_xyz[idx,:,:]=np.matmul(np.matmul(dx_dtle[idx,:],Cov_tle),dx_dtle[idx,:].T)
            frob_norms_pos[idx]=np.linalg.norm(Cov_xyz[idx,:3,:3],ord='fro') # Only position
            Cov_rtn_pos[idx,:,:]=np.matmul(np.matmul(transformation_matrix_cartesian_to_rtn, Cov_xyz[idx,:,:]),transformation_matrix_cartesian_to_rtn.T)[:3,:3]

        # Done propagating covariance
        cov_time_end = time.time()
        print("Finished in " + str(cov_time_end-cov_time_start) + " secs")
        
        # Create a list to save all propagated data
        print("\nWriting outputs to " + str(json_dir) + "...")
        write_time_start = time.time()

        # Precompute times for conversion
        times_str = [str(t) for t in times]

        # Run through all satellites
        for _i in range(len(self.tle_list)):
            # Current TLE
            current_tle = self.tle_list[_i]

            # Get starting index for state vectors
            start_indx = _i * len(times)

            # Propagate over time
            rTEME_arr = np.array([states_teme[start_indx + _j][0].detach().numpy() for _j in range(len(times))])
            vTEME_arr = np.array([states_teme[start_indx + _j][1].detach().numpy() for _j in range(len(times))]).T


            # Prepare CartesianDifferential
            velocity_TEME = CartesianDifferential(vTEME_arr * u.km / u.s)

            # Convert states to TEME frame and then to ITRS, GCRS, and Geodetic
            state_TEME_arr = [
                TEME(CartesianRepresentation(r * u.km, differentials=velocity_TEME[_j]), obstime=times[_j])
                for _j, r in enumerate(rTEME_arr)
            ]
            
            # Get states in ITRS and GCRS frames
            state_ITRS_arr = [state.transform_to(ITRS(obstime=times[_j])) for _j, state in enumerate(state_TEME_arr)]
            state_GCRS_arr = [state.transform_to(GCRS(obstime=times[_j])) for _j, state in enumerate(state_TEME_arr)]

            # Extract positions and velocities
            rGCRS_arr = np.array([state.cartesian.xyz.value for state in state_GCRS_arr])
            vGCRS_arr = np.array([state.velocity.d_xyz.value for state in state_GCRS_arr])

            # Convert to Geodetic
            lat_arr, lon_arr, height_arr = np.array([
                [loc.lat.deg, loc.lon.deg, loc.height.to(u.km).value]  # Access each component separately
                for loc in [EarthLocation.from_geocentric(state.x, state.y, state.z).to_geodetic() for state in state_ITRS_arr]
            ]).T

            # Save new position and velocity in ECI frame and Geodetic frame
            max_height = np.max(height_arr.tolist())
            min_height = np.min(height_arr.tolist())

            bulk_prop = {
                "name": current_tle._lines[0][2:],
                "Satellite catalog number": current_tle.satellite_catalog_number,
                "Classification": current_tle.classification,
                "International designator": current_tle.international_designator,
                "epoch": str(self.tle_epochs[_i]),
                "alt_max_km": max_height,
                "alt_min_km": min_height,
                "frontal_area_m2": (current_tle._bstar.numpy().tolist() * self.m / self.cd), # A = (B* m) / CD
                "time_utc": times_str,
                "position_eci_km": rGCRS_arr.tolist(),
                "velocity_eci_km_s": vGCRS_arr.tolist(),
                "latitude_deg": lat_arr.tolist(),
                "longitude_deg": lon_arr.tolist(),
                "height_deg": height_arr.tolist(),
                "covariance_position_rtn": Cov_rtn_pos[start_indx:(start_indx+len(times))].tolist(),
                "cov_fro_norm": frob_norms_pos[start_indx:(start_indx+len(times))].tolist()
            }

            # Save to json
            with open(json_dir + "/" + str(current_tle.satellite_catalog_number) + ".json", "w") as json_file:
                json.dump(bulk_prop, json_file, indent=4)
            self.json_tuple.append((str(current_tle.satellite_catalog_number),0,0,json.dumps(bulk_prop, indent=4)))
        write_time_end = time.time()
        print("Finished in " + str(write_time_end-write_time_start) + " secs")

if __name__ == "__main__":
    PUSH_TO_DATABASE = True
    DELETE_DATABASE = True

    # Connect to remote database
    conn = database.create_conn()
    cursor = conn.cursor()

    # Delete table from database
    if DELETE_DATABASE:
        database.delete_propagation_data(cursor=cursor)
    
    # Pull TLE data from database and parse
    tle = database.get_tle_data(cursor)
    tle_str_list = [t[1].split("\n") for t in tle]

    # Split up data
    for _i in range(0,1000,10):
        print("\n********************")
        print("Set: " + str(_i))
        print("********************")
        # set tle_list
        if _i == 0:
            tle_list_i = tle_str_list[-(_i+10):] 
        else:
            tle_list_i = tle_str_list[-(_i+10):-_i]  

        # Bulk import TLEs
        bulk = BulkTLE(
            tle_str_list=tle_list_i
        )

        # Set propagation times
        start_time = datetime(year=2025,month=2,day=16,hour=0,minute=0,second=0)
        datetime_list = [start_time + timedelta(minutes=i) for i in range(0,180,4)]

        # Bulk propagate
        bulk.bulk_propagate(
            times=datetime_list,
            json_dir="src/sgp4/sqp4_json"
        )

        # If you want to push the propagated data to the database
        if PUSH_TO_DATABASE:
            print("\nPushing Propagated Data to Database")

            # Upload data
            database.upload_propagation_data(
                cursor=cursor,
                propagation_data=bulk.json_tuple
            )

            # Commit
            conn.commit()