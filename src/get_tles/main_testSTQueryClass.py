#!/usr/bin/env python3

from src.get_tles.stQueryClass import stQueryClass


if __name__ == "__main__":
    uriBase                = "https://www.space-track.org"
    requestLogin           = "/ajaxauth/login"
    requestCmdAction       = "/basicspacedata/query" 
    requestFindLEOSats     = "/class/gp/EPOCH/>now-30/MEAN_MOTION/>11.25/format/3le"

    stq = stQueryClass("SLTrack.ini", uriBase + requestLogin, uriBase + requestCmdAction + requestFindLEOSats)
    fileName = stq.get3LELocal()
    print(fileName)