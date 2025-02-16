#!/usr/bin/env python3

from src.get_tles.stQueryClass import stQueryClass
import os

if __name__ == "__main__":
    uriBase                = "https://www.space-track.org"
    requestLogin           = "/ajaxauth/login"
    requestCmdAction       = "/basicspacedata/query" 
    requestFindLEOSats     = "/class/gp/EPOCH/>now-30/MEAN_MOTION/>11.25/format/3le"


    outputPath = os.getcwd()
    # print(outputPath)
    outputPath = os.path.join(outputPath, "src", "get_tles", "tleHistories")
    # print(outputPath)
    stq = stQueryClass("SLTrack.ini", uriBase + requestLogin, uriBase + requestCmdAction + requestFindLEOSats, outputPath)
    # fileName = stq.get3LELocal()
    # print(fileName)

    stq.writeST2DB()