#!/usr/bin/env python3

from src.get_tles.stQueryClass import stQueryClass
import os

def main():
    uriBase                = "https://www.space-track.org"
    requestLogin           = "/ajaxauth/login"
    requestCmdAction       = "/basicspacedata/query" 
    requestFindLEOSats     = "/class/gp/EPOCH/>now-30/MEAN_MOTION/>11.25/format/3le"

    outputPath = os.getcwd()
    outputPath = os.path.join(outputPath, "src", "get_tles", "tleHistories")
    stq = stQueryClass("SLTrack.ini", uriBase + requestLogin, uriBase + requestCmdAction + requestFindLEOSats, outputPath)

    stq.writeST2DB()
    stq.tle2keplerian()

if __name__ == "__main__":
    main()