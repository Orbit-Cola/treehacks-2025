import json

import src.utils.database as database

# Take advantage of interpreter caching. Only connect once.
conn = database.create_conn()

# Get propagator data
raw_propagator_data = database.get_propagation_data(conn.cursor())
if raw_propagator_data:
    json_data = json.loads(raw_propagator_data[0][3])
    PROPAGATOR_TIMESTEPS = json_data["time_utc"]
else:
    PROPAGATOR_TIMESTEPS = []
PROPAGATOR_DICT = {}
for obj in raw_propagator_data:
    json_data = json.loads(obj[3])
    PROPAGATOR_DICT[obj[0]] = json_data

# Get conjunction data
raw_conjunction_data = database.get_conjunctions_data(conn.cursor())
if raw_conjunction_data:
    json_data = json.loads(raw_conjunction_data[0][2])
CONJUNCTION_DICT = {}
for obj in raw_conjunction_data:
    json_data = json.loads(obj[2])
    CONJUNCTION_DICT[obj[0]] = json_data

# Get launches data
LAUNCHES_DATA = database.get_launches_data(conn.cursor())
