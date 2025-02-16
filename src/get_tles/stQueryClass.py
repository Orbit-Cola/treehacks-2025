import requests
# import json
import configparser
# import xlsxwriter
import time
from datetime import datetime
import glob
import os
from src.utils import database as db
from src.utils import credentials as cred

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
                 outputPath : str = "tleHistories",
                 QUERY_COOLDOWN = 3600):
        self.queryURL = queryURL
        self.loginURL = loginURL
        self.configFile = configFile
        self.outputPath = outputPath
        self.timeStamp = None
        self.QUERY_COOLDOWN = QUERY_COOLDOWN
        self.latestELSET = None
        self.parseConfigFile()

        onn = database.create_conn()
        cursor = conn.cursor()
        tle = database.get_tle_data(cursor)
        tle: list(tuple(satcat: str, tle: str))

    def parseConfigFile(self):
        configUsr = cred.USERNAME_ST
        configPwd = cred.PASSWORD_ST
        self.configOut = cred.OUTPUT_ST
        self.siteCred = {'identity': configUsr, 'password': configPwd}

    def readELSETFile(self, filepath):
        with open(filepath, "r") as f:
            textVar = f.read()
        return textVar
    
    def writeELSETFile(self, text, filepath):
        with open(filepath, "w") as f:
            f.write(text)

    def querySpaceTrack(self):
        with requests.Session() as session:
            # run the session in a with block to force session to close if we exit

            # need to log in first. note that we get a 200 to say the web site got the data, not that we are logged in
            resp = session.post(self.loginURL, data = self.siteCred)
            if resp.status_code != 200:
                raise MyError(resp, "POST fail on login")

            # this query picks up all Starlink satellites from the catalog. Note - a 401 failure shows you have bad credentials 
            resp = session.get(self.queryURL)
            if resp.status_code != 200:
                print(resp)
                raise MyError(resp, "GET fail on request for LEO satellites")
        
        print("Completed session")
        return resp.text
        
    def writeST2DB(self):
        ## UNFINISHED ## 
        now = datetime.now()
        db_time = 0

        if (now - db_time).total_seconds() < self.QUERY_COOLDOWN:
            print("Recent db data exists. Skipping query.")
        else:
            print("Cooldown has passed. Writing queried data to db")
            tleStr = self.querySpaceTrack()
            # self.latestELSET = tleStr

            strList = tleStr.splitlines()
            dbList = [ (strList[i], "\n".join(strList[i:i+3])) for i in range(0, len(strList) - 3, 3) ]


    def get3LELocal(self):
        # Define the filename pattern used previously
        file_pattern = self.configOut + "_*.txt"

        # Get the latest file matching the pattern
        files = glob.glob(os.path.join(self.outputPath, file_pattern))

        # Check if it is within the threshold
        now = datetime.now()

        if files:
            latest_file = max(files, key=os.path.getctime) # Get most recently created file
            latest_file_base = os.path.basename(latest_file)
            print(f"Latest file found: {latest_file_base}")

            # Extract timestamp from filename
            try:
                timestamp_str = latest_file_base.replace(self.configOut + "_", "").replace(".txt", "")
                file_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

                if (now - file_time).total_seconds() < self.QUERY_COOLDOWN:
                    print("Recent file exists. Skipping query and reading latest query.")
                    if self.latestELSET == None:
                        self.latestELSET = self.readELSETFile(latest_file)

                    return latest_file
                else:
                    print("Cooldown has passed. Let's run it back")
                    tleStr = self.querySpaceTrack()
                    self.latestELSET = tleStr
                    fullFilePath = os.path.join(self.outputPath, self.configOut + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
                    self.writeELSETFile(tleStr, fullFilePath)
                    return fullFilePath
                    
            except ValueError as e:
                print(e)
        
        else:
            print("No file found . Running query.")
            tleStr = self.querySpaceTrack()
            self.latestELSET = tleStr
            fullFilePath = os.path.join(self.outputPath, self.configOut + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
            self.writeELSETFile(tleStr, fullFilePath)
            return fullFilePath


        return self.latestELSET
