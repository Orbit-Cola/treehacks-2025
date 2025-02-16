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
