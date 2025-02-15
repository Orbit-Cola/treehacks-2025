from astropy.coordinates import TEME, ITRS, ICRS, CartesianDifferential, CartesianRepresentation, EarthLocation
from astropy.time import Time
import astropy.units as u
import numpy as np
from sgp4.api import Satrec, jday, accelerated, days2mdhms
from sgp4.conveniences import sat_epoch_datetime
from datetime import datetime, timedelta
import json
from tqdm import tqdm
# from dsgp4 
# from valladopy.astro import sgp4 TODO implement official vallado version

# Try maybe faster version
import sys
import os

# Add the parent directory to the system path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python-sgp4-the-return-of-numba'))

# from sgp4_faster.api import Satrec, jday, accelerated, days2mdhms
# from sgp4_faster.conveniences import sat_epoch_datetime

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

class TLE:
    """
    Class to hold a TLE object
    """
    def __init__(self,sat_name:str,line1:str,line2:str):
        """
        TLE class constructor function

        Parameters
        ----------
        sat_name: string
            The satellite's name
        line1: string
            The first line of the TLE e.g. "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
        line2: string
            The second line of the TLE e.g. "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
        """
        # Create class variables
        self.sat_name = sat_name
        self.line1 = line1
        self.line2 = line2

        # Extract epoch
        satellite = Satrec.twoline2rv(self.line1,self.line2)
        month, day, hour, minute, second = days2mdhms(satellite.epochyr,satellite.epochdays)
        year = sat_epoch_datetime(satellite).year
        microseconds = int((second - int(second)) * 1e6)
        self.epoch = datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=int(second),
            microsecond=microseconds
        )

        # TODO implement parameters
        self.frontal_area = 0.
        self.id = -1.
        self.covariance = np.zeros((6,6))

        # TLE variables (https://celestrak.org/NORAD/documentation/tle-fmt.php)
        self.satellite_number = None # TODO
        self.classification = None # TODO
        self.launch_year = None # TODO
        self.launch_number = None # TODO
        self.launch_piece = None # TODO
        self.dt_mean_motion = None # TODO
        self.dt2_mean_motion = None # TODO
        self.bstar = None # TODO
        self.ephemeris_type = None # TODO
        self.element_num = None # TODO
        self.inclination = None # TODO
        self.right_ascension_ascending_node = None # TODO
        self.eccentricity = None # TODO
        self.arg_of_perigee = None # TODO
        self.mean_anomaly = None # TODO
        self.mean_motion = None # TODO
        self.rev_at_epoch = None # TODO

def propagate_tle(tle:TLE,prop_date:datetime):
    """
    Propagate a satellite's TLE using the SGP4 propagator from the python sgp4 library to a desired date.

    Parameters
    ----------
    tle: TLE
        TLE class object containing the satellite's TLE
    prop_date: datetime
        The date and time to propagate the satellite's orbit to
    """
    # Build satellite from TLE
    satellite = Satrec.twoline2rv(tle.line1,tle.line2)

    # Get the julian date
    jd,fr = jday(
        year=prop_date.year,
        mon=prop_date.month,
        day=prop_date.day,
        hr=prop_date.hour,
        minute=prop_date.minute,
        sec=prop_date.second
    )

    # Propagate TLE to desired date
    sgp4_error,rTEME_tuple,vTEME_tuple = satellite.sgp4(jd,fr)

    # Check for errors in propagation
    if not sgp4_error:# Get Astropy Time object to perform frame conversions
        obstime = Time(jd+fr,format='jd')
        
        # Create astropy TEME frame objects
        rTEME_array = np.array(rTEME_tuple) * u.km
        vTEME_array = np.array(vTEME_tuple) * u.km / u.s
        state_TEME = TEME(
            CartesianRepresentation(rTEME_array,differentials=CartesianDifferential(vTEME_array)),
            obstime=obstime
        )
        rTEME = state_TEME.cartesian.xyz.value
        vTEME = state_TEME.velocity.d_xyz.value

        # Get state as SatelliteState
        sat_state = SatelliteState(
            rTEME=rTEME,
            vTEME=vTEME,
            timestamp=prop_date
        )
    else:
        sat_state = None
    
    return sat_state, sgp4_error

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
        # Build a list to store all TLE objects
        self.tle_list = []

        # Open text file and record lines
        with open(tle_txt, "r") as file:
            lines = file.readlines()

        # Extract every TLE set
        for i in tqdm(range(0, len(lines), 3), desc="Bulk Reading TLEs", unit="TLE"):
            sat_name = lines[i].strip()[2:]
            tle_line1 = lines[i + 1].strip()
            tle_line2 = lines[i + 2].strip()
            
            # Add all TLE objects to the list
            self.tle_list.append(TLE(
                sat_name=sat_name,
                line1=tle_line1,
                line2=tle_line2
            ))
    
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
        # Create a list to save all propagated data
        bulk_prop = []

        # Run through each satellite
        for _tle in tqdm(self.tle_list, desc="Bulk Propagating TLEs", unit="TLE"):
            position_eci = [] # [km]
            velocity_eci = [] # [km/s]
            position_geodetic = [] # [rad,rad,km]
            heights = []
            times_str = []

            # Propagate over time
            for _t in times:
                # Log timestamp as string
                times_str.append(str(_t))

                # Propagate to timestamp
                sat_state, sgp4_error = propagate_tle(
                    tle=_tle,
                    prop_date=_t
                )

                if not sgp4_error:
                    # Add ECI positions and velocities
                    position_eci.append(sat_state.rICRS.tolist())
                    velocity_eci.append(sat_state.vICRS.tolist())

                    # Add Geodetic coords
                    position_geodetic.append(sat_state.rGeodetic.tolist())
                    heights.append(sat_state.rGeodetic[2])

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
                "name": _tle.sat_name,
                "id": _tle.id,
                "epoch": str(_tle.epoch),
                "alt_max_km": max_height,
                "alt_min_km": min_height,
                "frontal_area_m2":_tle.frontal_area,
                "time_utc": times_str,
                "position_eci_km": position_eci,
                "velocity_eci_km_s": velocity_eci,
                "position_geodetic_rad_km": position_geodetic,
            })
        
        # Save to json
        with open(prop_json, "w") as json_file:
            json.dump(bulk_prop, json_file, indent=4)  # Pretty-print with indentation

if __name__ == "__main__":
    bulk = BulkTLE(
        tle_txt="src/sgp4/tle.txt"
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