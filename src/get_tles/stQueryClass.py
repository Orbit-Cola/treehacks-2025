import requests
# import json
import configparser
# import xlsxwriter
import time
from datetime import datetime
import glob
import os

# GM = 398600441800000.0
# GM13 = GM ** (1.0/3.0)
# MRAD = 6378.137
# PI = 3.14159265358979
# TPI86 = 2.0 * PI / 86400.0

class MyError(Exception):
    def __init__(self,args):
        Exception.__init__(self,"my exception was raised with arguments {0}".format(args))
        self.args = args

class stQueryClass():
    def __init__(self, 
                 configFile : str,
                 loginURL : str, 
                 queryURL : str,
                 QUERY_COOLDOWN = 3600):
        self.queryURL = queryURL
        self.loginURL = loginURL
        self.configFile = configFile
        self.timeStamp = None
        self.QUERY_COOLDOWN = QUERY_COOLDOWN
        self.parseConfigFile()

    def parseConfigFile(self):
        config = configparser.ConfigParser()
        config.read("./" + self.configFile)
        configUsr = config.get("configuration","username")
        configPwd = config.get("configuration","password")
        self.configOut = config.get("configuration","output")
        self.siteCred = {'identity': configUsr, 'password': configPwd}

    def queryST(self):
        # Define the filename pattern used previously
        file_pattern = self.configOut + "_*.txt"

        # Get the latest file matching the pattern
        files = glob.glob(file_pattern)
        if files:
            latest_file = max(files, key=os.path.getctime)  # Get most recently created file
            print(f"Latest file found: {latest_file}")

            # Extract timestamp from filename
            try:
                timestamp_str = latest_file.split("_")[1].replace(".txt", "")  # Extract time part
                file_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

                # Check if it is within the threshold
                now = datetime.now()
                if (now - file_time).total_seconds() < self.QUERY_COOLDOWN:
                    print("Recent file exists. Skipping query.")
                else:
                    print("File is too old. Running query.")
                    with requests.Session() as session:
                        # run the session in a with block to force session to close if we exit

                        # need to log in first. note that we get a 200 to say the web site got the data, not that we are logged in
                        resp = session.post(self.loginURL)
                        if resp.status_code != 200:
                            raise MyError(resp, "POST fail on login")

                        # this query picks up all Starlink satellites from the catalog. Note - a 401 failure shows you have bad credentials 
                        resp = session.get(self.queryURL)
                        if resp.status_code != 200:
                            print(resp)
                            raise MyError(resp, "GET fail on request for LEO satellites")

                        with open(self.configOut + "_" + now.strftime("%Y-%m-%d_%H-%M-%S"), "w") as f:
                            f.write(resp.text)

                    print("Completed session")
            except ValueError:
                print("Error: Filename timestamp format incorrect.")
        else:
            print("No previous file found. Running query.")