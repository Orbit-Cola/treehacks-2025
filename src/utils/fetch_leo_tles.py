import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.st_query import STQuery

if __name__ == "__main__":
    url_base = "https://www.space-track.org"
    request_login = "/ajaxauth/login"
    request_cmd_action = "/basicspacedata/query" 
    request_find_leo_sats = "/class/gp/EPOCH/>now-30/MEAN_MOTION/>11.25/format/3le"
    stq = STQuery(
        "SLTrack.ini",
        url_base + request_login,
        url_base + request_cmd_action + request_find_leo_sats,
    )
    stq.write_to_database()
    stq.tle2keplerian()
