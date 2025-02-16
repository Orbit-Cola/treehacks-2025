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
# from src.get_tles.stQueryClass import stQueryClass

# class SatelliteCovariance():
#     """
#     Class to hold the satellite covariance matrix in different frames
#     TODO accept covariances from different frames
#     """
#     def __init__(
#             self,
#             covRIC:np.ndarray,
#             state:SatelliteState
#     ):
#         # Create class variables
#         self.state = state              # The state of the object corresponding to the covariance matrix
#         self.covRIC = covRIC            # Covariance matrix in RIC frame
#         self.covICRS = np.zeros((6,6))  # Covariance matrix in ICRS frame
#         self.covITRS = np.zeros((6,6))  # Covariance matrix in ITRS frame

#         # Get covariance in ICRS and ITRS frame
#         self._covariance_ric2icrs()
#         self._covariance_ric2itrs()
    
#     def _covariance_ric2icrs(
#             self,
#     ):
#         """
#         Convert covariance matrix from RIC frame to ICRS frame
#         """
#         # Get the covariance matrix
#         self.covICRS = self.state.ICRS2RIC.T @ self.covRIC @ self.state.ICRS2RIC
    
#     def _covariance_ric2itrs(
#             self,
#     ):
#         """
#         Convert covariance matrix from RIC frame to ITRS frame
#         """
#         # Get the covariance matrix
#         self.covITRS = self.state.ITRS2RIC.T @ self.covRIC @ self.state.ITRS2RIC
    
#     def propagate_covariance(
#             self,
#             stm_icrs:np.ndarray,
#             propagated_state:SatelliteState,
#     ):
#         """
#         Given a state transition matrix (STM) in the ICRS frame and the propagated state, propagate
#         the covariance matrix

#         Parameters
#         ----------
#         stm_icrs: numpy.ndarray
#             The STM in the ICRS frame
#         propagated_state: SatelliteState
#             The propagated state corresponding to the propagated covariance matrix

#         Returns
#         -------
#         propagated_cov_obj: SatelliteCovariance
#             The propagated covariance matrix
#         """
#         # Get new covariance matrix in ICRS
#         propagated_covICRS = stm_icrs @ self.covICRS @ stm_icrs.T

#         # Build new SatelliteCovariance object
#         propagated_cov_obj = self.build_from_covICRS(covICRS=propagated_covICRS,state=propagated_state)

#         return propagated_cov_obj
    
#     @staticmethod
#     def build_from_covICRS(
#         covICRS:np.ndarray,
#         state:SatelliteState,
#     ):
#         """
#         Build a SatelliteCovariance object from a covariance matrix in the ICRS frame

#         Parameters
#         ----------
#         covICRS: numpy.ndarray
#             The covariance matrix in the ICRS frame
#         state: SatelliteState
#             The state corresponding to the covariance matrix
        
#         Returns
#         -------
#         cov_obj: SatelliteCovariance
#             The covariance object
#         """
#         # Convert ICRS to RIC
#         covRIC = state.ICRS2RIC @ covICRS @ state.ICRS2RIC.T

#         # Build new SatelliteCovariance object
#         cov_obj = SatelliteCovariance(covRIC=covRIC,state=state)

#         return cov_obj

class BulkTLE:
    """
    Class to bulk propagate TLE data
    """
    def __init__(self,tle_txt:str):
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

        # Open text file and record all TLEs
        self.tle_list = []
        tle_list_init=dsgp4.tle.load(tle_txt)

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
            state (`numpy.array`): numpy array of 2 rows and 3 columns, where
                                        the first row represents position, and the second velocity.
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
    
    def bulk_propagate(self,times:list,prop_json:str):
        """
        Bulk propagate the self.tle_list of TLEs over the list of input times

        Parameters
        ----------
        times: list
            List of desired datetimes as datetime
        prop_json: str
            The file to store propagated data as json
        """
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
        #this is the initial TLE covariance matrix:
        Cov_tle=np.array([[ 1.06817079e-23,  8.85804989e-25,  1.51328946e-24,
                -2.48167092e-13,  1.80784129e-11, -9.17516946e-17,
                -1.80719145e-11,  2.47782854e-14,  1.06374440e-19],
            [ 1.10888880e-26,  9.19571327e-28,  1.57097512e-27,
                -2.57618033e-16,  1.87675528e-14, -9.28440729e-20,
                -1.87608028e-14,  2.57219640e-17,  1.10539220e-22],
            [-3.62208982e-24, -3.00370060e-25, -5.13145502e-25,
                8.41515501e-14, -6.13025898e-12,  3.11161104e-17,
                6.12805538e-12, -8.40212641e-15, -3.61913778e-20],
            [-2.72347076e-13, -2.25849762e-14, -3.85837006e-14,
                6.69613552e-03, -4.60858400e-01,  2.44529381e-06,
                4.60848714e-01, -6.66633934e-04,  2.73554382e-10],
            [ 1.98398791e-11,  1.64526723e-12,  2.81073778e-12,
                -4.60858400e-01,  3.35783115e+01, -1.70385711e-04,
                -3.35662070e+01,  4.60149046e-02,  1.68286334e-07],
            [-1.00692291e-16, -8.34937429e-18, -1.42650203e-17,
                2.44529381e-06, -1.70385711e-04,  1.75093676e-05,
                1.70379140e-04, -2.42796071e-07, -2.62336921e-10],
            [-1.98327475e-11, -1.64467582e-12, -2.80972744e-12,
                4.60848714e-01, -3.35662070e+01,  1.70379140e-04,
                3.35541747e+01, -4.60131157e-02, -2.28248504e-07],
            [ 2.71925407e-14,  2.25500097e-15,  3.85239630e-15,
                -6.66633934e-04,  4.60149046e-02, -2.42796071e-07,
                -4.60131157e-02,  6.63783423e-05, -3.21657063e-12],
            [ 1.16764843e-19,  9.73507662e-21,  1.65971417e-20,
                2.73554368e-10,  1.68286335e-07, -2.62336921e-10,
                -2.28248505e-07, -3.21656925e-12,  2.21029182e-07]])/1000.
        frob_norms_pos=np.zeros((len(tle_batch_list),))

        dx_dtle = torch.zeros((len(tsinces),6,9))
        for k in range(len(tsinces)):
            for i in range(6):
                tle_elements[k].grad=None
                states_teme[k].flatten()[i].backward(retain_graph=True)
                dx_dtle[k,i,:] = tle_elements[k].grad
        
        for idx, tle in enumerate(tle_batch_list):
            state_rtn, cartesian_to_rtn_rotation_matrix = self.from_cartesian_to_rtn(states_teme[idx].detach().numpy())
            # I construct the 6x6 rotation matrix from cartesian -> RTN
            transformation_matrix_cartesian_to_rtn = np.zeros((6,6))
            transformation_matrix_cartesian_to_rtn[0:3, 0:3] = cartesian_to_rtn_rotation_matrix
            transformation_matrix_cartesian_to_rtn[3:,3:] = cartesian_to_rtn_rotation_matrix

            Cov_xyz[idx,:,:]=np.matmul(np.matmul(dx_dtle[idx,:],Cov_tle),dx_dtle[idx,:].T)
            frob_norms_pos[idx]=np.linalg.norm(Cov_xyz[idx,:3,:3],ord='fro')
            Cov_rtn_pos[idx,:,:]=np.matmul(np.matmul(transformation_matrix_cartesian_to_rtn, Cov_xyz[idx,:,:]),transformation_matrix_cartesian_to_rtn.T)[:3,:3]
            # TODO only position
            # TODO 1 json each sat
        
        # Done propagating covariance
        cov_time_end = time.time()
        print("Finished in " + str(cov_time_end-cov_time_start) + " secs")
        
        # Create a list to save all propagated data
        print("\nWriting outputs to " + str(prop_json) + "...")
        write_time_start = time.time()
        bulk_prop = []

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

            bulk_prop.append({
                "name": current_tle._lines[0][2:],
                "Satellite catalog number": current_tle.satellite_catalog_number,
                "Classification": current_tle.classification,
                "International designator": current_tle.international_designator,
                "epoch": str(self.tle_epochs[_i]),
                "alt_max_km": max_height,
                "alt_min_km": min_height,
                "frontal_area_m2": current_tle._bstar.numpy().tolist(),
                "time_utc": times_str,
                "position_eci_km": rGCRS_arr.tolist(),
                "velocity_eci_km_s": vGCRS_arr.tolist(),
                "latitude_deg": lat_arr.tolist(),
                "longitude_deg": lon_arr.tolist(),
                "height_deg": height_arr.tolist(),
                "covariance_position_rtn": Cov_rtn_pos[start_indx:(start_indx+len(times))].tolist(),
                "cov_fro_norm": frob_norms_pos[start_indx:(start_indx+len(times))].tolist()
            })

        
        # Save to json
        with open(prop_json, "w") as json_file:
            json.dump(bulk_prop, json_file, indent=4)
        write_time_end = time.time()
        print("Finished in " + str(write_time_end-write_time_start) + " secs")

if __name__ == "__main__":
    # uriBase                = "https://www.space-track.org"
    # requestLogin           = "/ajaxauth/login"
    # requestCmdAction       = "/basicspacedata/query" 
    # requestFindLEOSats     = "/class/gp/EPOCH/>now-30/MEAN_MOTION/>11.25/format/3le"

    # stq = stQueryClass("SLTrack.ini", uriBase + requestLogin, uriBase + requestCmdAction + requestFindLEOSats)
    # fileName = stq.get3LELocal()

    bulk = BulkTLE(
        tle_txt="src/sgp4/tle_test.txt"#"src/get_tles/tleHistories/tleHistory_2025-02-15_16-08-15.txt"
    )
    prop_date1 = datetime(
        year=2025,
        month=2,
        day=15,
        hour=0,
        minute=0,
        second=0
    )
    bulk.bulk_propagate(
        times=[prop_date1],
        prop_json="src/sgp4/sqp4_test.json"
    )