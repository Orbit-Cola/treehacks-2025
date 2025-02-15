from astropy.coordinates import TEME, ITRS, ICRS, CartesianDifferential, CartesianRepresentation, EarthLocation
import astropy.units as u
import numpy as np
from datetime import datetime, timedelta
import json
import dsgp4
import torch
import time

class SatelliteState():
    """
    Class to hold the satellite states in different frames
    TODO accept position vectors in many different frames
    """
    def __init__(
            self,
            rTEME:np.ndarray,
            vTEME:np.ndarray,
            timestamp:datetime
    ):
        # Create class variables
        self.rTEME = rTEME            # Position vector TEME frame (km)
        self.vTEME = vTEME            # Velocity vector TEME frame (km/s)
        self.rICRS = np.zeros(3)      # Position vector ICRS (ECI) frame (km)
        self.vICRS = np.zeros(3)      # Velocity vector ICRS (ECI) frame (km/s)
        self.rITRS = np.zeros(3)      # Position vector ITRS (ECEF) frame (km)
        self.vITRS = np.zeros(3)      # Velocity vector ITRS (ECEF) frame (km/s)
        self.rGeodetic = np.zeros(3)  # Position vector Geodetic frame [rad,rad,km]
        self.timestamp = timestamp    # Timestamp of data point
        self.ICRS2RIC = np.zeros((3,3)) # Rotation matrix from ICRS (ECI) to RIC frame
        self.ITRS2RIC = np.zeros((3,3)) # Rotation matrix from ITRS (ECEF) to RIC frame

        # Get states to TEME frame
        self.rTEME = self.rTEME * u.km
        self.vTEME = self.vTEME * u.km / u.s
        state_TEME = TEME(
            CartesianRepresentation(self.rTEME,differentials=CartesianDifferential(self.vTEME)),
            obstime=self.timestamp
        )
        self.rTEME = state_TEME.cartesian.xyz.value
        self.vTEME = state_TEME.velocity.d_xyz.value
        
        # Convert to ITRS
        state_ITRS = state_TEME.transform_to(ITRS(obstime=self.timestamp))
        self.rITRS = state_ITRS.cartesian.xyz.value
        self.vITRS = state_ITRS.velocity.d_xyz.value

        # Convert to ICRS
        state_ICRS = state_TEME.transform_to(ICRS())
        self.rICRS = state_ICRS.cartesian.xyz.value
        self.vICRS = state_ICRS.velocity.d_xyz.value

        # Convert to Geodetic
        location = EarthLocation.from_geocentric(state_ITRS.x,state_ITRS.y,state_ITRS.z)
        lat, lon, alt = location.to_geodetic()
        self.rGeodetic = np.array([lat.deg, lon.deg, alt.to(u.km).value])

        # Get rotation matrices
        self.ICRS2RIC = self._icrs2ric()
        self.ITRS2RIC = self._itrs2ric()
    
    def _icrs2ric(
            self,
    ):
        """
        Get the rotation matrix between the ICRS and RIC frames. To get the opposite rotation, take the transpose.
        The rotation matrix should be orthogonal.
        """
        # Get RIC from ICRS
        r_dir = self.rICRS / np.linalg.norm(self.rICRS)
        c_dir = np.cross(self.rICRS,self.vICRS) / np.linalg.norm(np.cross(self.rICRS,self.vICRS))
        i_dir = np.cross(c_dir,r_dir)

        # Build rotation matrix
        self.ICRS2RIC = np.array([r_dir,i_dir,c_dir])
    
    def _itrs2ric(
            self,
    ):
        """
        Get the rotation matrix between the ITRS and RIC frames. To get the opposite rotation, take the transpose.
        The rotation matrix should be orthogonal.
        """
        # Get RIC from ICRS
        r_dir = self.rITRS / np.linalg.norm(self.rITRS)
        c_dir = np.cross(self.rITRS,self.vITRS) / np.linalg.norm(np.cross(self.rITRS,self.vITRS))
        i_dir = np.cross(c_dir,r_dir)

        # Build rotation matrix
        self.ITRS2RIC = np.array([r_dir,i_dir,c_dir])

class SatelliteCovariance():
    """
    Class to hold the satellite covariance matrix in different frames
    TODO accept covariances from different frames
    """
    def __init__(
            self,
            covRIC:np.ndarray,
            state:SatelliteState
    ):
        # Create class variables
        self.state = state              # The state of the object corresponding to the covariance matrix
        self.covRIC = covRIC            # Covariance matrix in RIC frame
        self.covICRS = np.zeros((6,6))  # Covariance matrix in ICRS frame
        self.covITRS = np.zeros((6,6))  # Covariance matrix in ITRS frame

        # Get covariance in ICRS and ITRS frame
        self._covariance_ric2icrs()
        self._covariance_ric2itrs()
    
    def _covariance_ric2icrs(
            self,
    ):
        """
        Convert covariance matrix from RIC frame to ICRS frame
        """
        # Get the covariance matrix
        self.covICRS = self.state.ICRS2RIC.T @ self.covRIC @ self.state.ICRS2RIC
    
    def _covariance_ric2itrs(
            self,
    ):
        """
        Convert covariance matrix from RIC frame to ITRS frame
        """
        # Get the covariance matrix
        self.covITRS = self.state.ITRS2RIC.T @ self.covRIC @ self.state.ITRS2RIC
    
    def propagate_covariance(
            self,
            stm_icrs:np.ndarray,
            propagated_state:SatelliteState,
    ):
        """
        Given a state transition matrix (STM) in the ICRS frame and the propagated state, propagate
        the covariance matrix

        Parameters
        ----------
        stm_icrs: numpy.ndarray
            The STM in the ICRS frame
        propagated_state: SatelliteState
            The propagated state corresponding to the propagated covariance matrix

        Returns
        -------
        propagated_cov_obj: SatelliteCovariance
            The propagated covariance matrix
        """
        # Get new covariance matrix in ICRS
        propagated_covICRS = stm_icrs @ self.covICRS @ stm_icrs.T

        # Build new SatelliteCovariance object
        propagated_cov_obj = self.build_from_covICRS(covICRS=propagated_covICRS,state=propagated_state)

        return propagated_cov_obj
    
    @staticmethod
    def build_from_covICRS(
        covICRS:np.ndarray,
        state:SatelliteState,
    ):
        """
        Build a SatelliteCovariance object from a covariance matrix in the ICRS frame

        Parameters
        ----------
        covICRS: numpy.ndarray
            The covariance matrix in the ICRS frame
        state: SatelliteState
            The state corresponding to the covariance matrix
        
        Returns
        -------
        cov_obj: SatelliteCovariance
            The covariance object
        """
        # Convert ICRS to RIC
        covRIC = state.ICRS2RIC @ covICRS @ state.ICRS2RIC.T

        # Build new SatelliteCovariance object
        cov_obj = SatelliteCovariance(covRIC=covRIC,state=state)

        return cov_obj

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
        print("\nPruned by Deep Space: " + str(len(tle_list_init)-len(self.tle_list)))

        # Extract all epochs
        self.tle_epochs = []
        for _tle in self.tle_list:
            self.tle_epochs.append(datetime(year=_tle.epoch_year,month=1,day=1) + timedelta(days=_tle.epoch_days))
        
        # Report number of TLEs
        print("\nNumber of TLEs: " + str(len(self.tle_list)))
    
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
            tsince_list = [(dt - tle_epoch).total_seconds() / 60 for dt in times]
            tsinces.append(tsince_list)

        # Convert to a tensor for batch processing
        tsinces_tensor = torch.tensor(tsinces).flatten()
        print(tsinces_tensor)
        
        # Initialize TLE batch
        _,tle_batch=dsgp4.initialize_tle(tle_batch_list)

        # Propagate batch
        print("\nDesired Propagation Timestamps: " + str(len(times)))
        print("\nRunning TLE Batch Propagation...")
        prop_time_start = time.time()
        states_teme = dsgp4.propagate_batch(tle_batch,tsinces_tensor)
        prop_time_end = time.time()
        print("Finished in " + str(prop_time_end-prop_time_start) + " secs")

        # Create a list to save all propagated data
        print("\nWriting outputs to " + str(prop_json) + "...")
        write_time_start = time.time()
        bulk_prop = []

        # Run through each satellite
        for _i in range(len(self.tle_list)):
            # Current TLE
            current_tle = self.tle_list[_i]

            # Position, velocity, height, times
            position_eci = [] # [km]
            velocity_eci = [] # [km/s]
            position_geodetic = [] # [rad,rad,km]
            heights = []
            times_str = []
            
            # Get starting index for state vectors
            start_indx = _i*len(times)

            # Propagate over time
            for _j in range(len(times)):
                # Log timestamp as string
                times_str.append(str(times[_j]))

                # Get state and velocity TEME
                rTEME = states_teme[start_indx+_j][0].numpy()
                vTEME = states_teme[start_indx+_j][1].numpy()
                
                # Get states to TEME frame
                state_TEME = TEME(
                    CartesianRepresentation(rTEME * u.km,differentials=CartesianDifferential(vTEME * u.km / u.s)),
                    obstime=times[_j]
                )
                
                # Convert to ITRS
                state_ITRS = state_TEME.transform_to(ITRS(obstime=times[_j]))
                rITRS = state_ITRS.cartesian.xyz.value
                vITRS = state_ITRS.velocity.d_xyz.value

                # Convert to ICRS
                state_ICRS = state_TEME.transform_to(ICRS())
                rICRS = state_ICRS.cartesian.xyz.value
                vICRS = state_ICRS.velocity.d_xyz.value

                # Convert to Geodetic
                location = EarthLocation.from_geocentric(state_ITRS.x,state_ITRS.y,state_ITRS.z)
                lat, lon, alt = location.to_geodetic()
                rGeodetic = np.array([lat.deg, lon.deg, alt.to(u.km).value])
                    
                # Add ECI positions and velocities
                position_eci.append(rICRS.tolist())
                velocity_eci.append(vICRS.tolist())

                # Add Geodetic coords
                position_geodetic.append(rGeodetic.tolist())
                heights.append(rGeodetic[2])

            # Save new position and velocity in ECI frame and Geodetic frame
            max_height = -1.
            min_height = -1.
            try: 
                max_height = max(heights)
                min_height = min(heights)
            except:
                max_height = -1.
                min_height = -1.
            bulk_prop.append({
                "name": current_tle._lines[0][2:],
                "Satellite catalog number": current_tle.satellite_catalog_number,
                "Classification": current_tle.classification,
                "International designator": current_tle.international_designator,
                "epoch": str(self.tle_epochs[_i]),
                "alt_max_km": max_height,
                "alt_min_km": min_height,
                "frontal_area_m2":current_tle._bstar.numpy().tolist(),
                "time_utc": times_str,
                "position_eci_km": position_eci,
                "velocity_eci_km_s": velocity_eci,
                "position_geodetic_rad_km": position_geodetic,
            })
        
        # Save to json
        with open(prop_json, "w") as json_file:
            json.dump(bulk_prop, json_file, indent=4)
        write_time_end = time.time()
        print("Finished in " + str(write_time_end-write_time_start) + " secs")

if __name__ == "__main__":
    bulk = BulkTLE(
        tle_txt="src/sgp4/tle_test.txt"
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
        prop_json="test.json"
    )