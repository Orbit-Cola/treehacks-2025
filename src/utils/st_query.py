from datetime import datetime, timedelta
import dsgp4
import requests

from src.utils import database
from src.utils import credentials as creds

GM = 398600.441800000
MRAD = 6378.137

class MyError(Exception):
    def __init__(self,args):
        Exception.__init__(self,"my exception was raised with arguments {0}".format(args))
        self.args = args

def epoch2time(epoch_year, epoch_day):
    yy = int(epoch_year)
    dd = float(epoch_day)
    base_date = datetime(yy, 1, 1)  # Start of the year
    tle_datetime = base_date + timedelta(days=dd - 1)  # Subtract 1 since Jan 1st is day 1
    return tle_datetime.strftime("%Y-%m-%d %H:%M:%S")

def mean_motion2sma(n):
    n = float(n)
    n /= 60
    return (GM / (n ** 2)) ** (1/3)

class STQuery():
    def __init__(self, 
                 config_file : str,
                 login_url : str, 
                 query_url : str,
                 QUERY_COOLDOWN = 3600):
        self.query_url = query_url
        self.login_url = login_url
        self.config_file = config_file
        self.timestamp = None
        self.QUERY_COOLDOWN = QUERY_COOLDOWN
        self.parse_config_file()

        self.conn = database.create_conn()

    def parse_config_file(self):
        self.site_creds = {'identity': creds.USERNAME_ST, 'password': creds.PASSWORD_ST}


    def query_space_track(self):
        with requests.Session() as session:
            # Need to log in first. Note that we get a 200 to say the web site got the data, not that we are logged in.
            resp = session.post(self.login_url, data = self.site_creds)
            if resp.status_code != 200:
                raise MyError(resp, "POST fail on login")
            # Note: a 401 failure means you have bad credentials.
            resp = session.get(self.query_url)
            if resp.status_code != 200:
                print(resp)
                raise MyError(resp, "GET fail on request for LEO satellites")
        
        print("Completed session")
        return resp.text
        
    def write_to_database(self):
        now = datetime.now()
        cursor = self.conn.cursor()
        db_time_list = database.get_latest_tle_upload_timestamp(cursor)
        if not db_time_list or (now - db_time_list[0][0]).total_seconds() > self.QUERY_COOLDOWN:
            print("Cooldown has passed. Writing queried data to db.")
            tle_str = self.query_space_track()
            str_list = tle_str.splitlines()
            db_list = [(str_list[i + 1].split()[1], "\n".join(str_list[i:i + 3])) for i in range(0, len(str_list) - 3, 3)]
            database.upload_tle_data(cursor, db_list)
            database.upload_tle_upload_timestamp(cursor, (now.strftime("%Y-%m-%d %H:%M:%S"),))
            self.conn.commit()
        else:
            print("Recent db data exists. Skipping query.")

    def tle2keplerian(self):
        cursor = self.conn.cursor()
        tle_list = [dsgp4.tle.TLE(t[1].splitlines()) for t in database.get_tle_data(cursor)]
        db_list = [
            (
                tle.satellite_catalog_number,
                mean_motion2sma(tle._no_kozai),
                float(tle._ecco),
                float(tle._inclo),
                float(tle._nodeo),
                float(tle._argpo),
                epoch2time(tle.epoch_year, float(tle.epoch_days))
            )
            for tle in tle_list
        ]
        database.upload_keplerian_data(cursor, db_list)
        self.conn.commit()
        print("Keplerian elements converted and uploaded from TLEs.")
