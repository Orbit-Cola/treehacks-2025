from datetime import datetime, timedelta
import requests
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils.database as database

def get_results(query_url):
    try:
        results = requests.get(query_url)
    except Exception as e:
        print(f"Exception: {e}")
    else:
        status = results.status_code
        print(f"Status code: {status}")
        if status != 200:
            return
        return results.json()

launch_base_url = "https://ll.thespacedevs.com/2.3.0/launches/"
now = datetime.now()
month_ahead = now + timedelta(days=31)
filters = f"net__lte={month_ahead.isoformat()}"
limit = "limit=100"
ordering = "ordering=-net"
query_url = launch_base_url + "?" + "&".join((filters, limit, ordering))
results = get_results(query_url)

data = []
for launch in results["results"]:
    name = launch["name"]
    net = datetime.strptime(launch["net"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
    window_end = datetime.strptime(launch["window_end"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
    window_start = datetime.strptime(launch["window_start"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
    lsp = launch["launch_service_provider"]["name"]
    rocket = launch["rocket"]["configuration"]["full_name"]
    data.append((name, net, window_end, window_start, lsp, rocket))

conn = database.create_conn()
database.upload_launches(conn.cursor(), data)
conn.commit()
