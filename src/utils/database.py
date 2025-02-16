import mysql.connector
import sshtunnel

from src.utils.credentials import *

########
# HOW-TO:
# import database
# conn = database.create_conn()
# cursor = conn.cursor()
# tle = database.get_tle_data(cursor)
# tle: list(tuple(satcat: str, tle: str))
########

def create_conn():
    tunnel = sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com', 22),
        ssh_username=USERNAME, ssh_password=PA_PASSWORD,
        remote_bind_address=(DB_HOSTNAME, 3306)
    )
    tunnel.start()
    conn = mysql.connector.connect(
        user=USERNAME,
        password=DB_PASSWORD,
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        database=DB_NAME,
        use_pure=True,
    )
    return conn

def upload_tle_data(cursor, tle_data):
    # Data should be list of tuple(satcat: str, tle: str)
    cursor.execute("DELETE FROM tle")  # Delete all old records
    insert_tle = "INSERT INTO tle(satcat, tle) VALUES(%s, %s)"
    cursor.executemany(insert_tle, tle_data)

def upload_tle_upload_timestamp(cursor, stamp):
    insert_timestamp = "INSERT INTO tle_upload(stamp) VALUES(%s)"
    cursor.execute(insert_timestamp, stamp)

def delete_propagation_data(cursor):
    cursor.execute("DELETE FROM propagation")  # Delete all old records

def upload_propagation_data(cursor, propagation_data):
    # Data should be list of tuple(satcat: str, apogee_km: float, perigee_km: float, data: str (serialized JSON))
    insert_propagation = "INSERT INTO propagation(satcat, apogee_km, perigee_km, data) VALUES(%s, %s, %s, %s)"
    cursor.executemany(insert_propagation, propagation_data)

def upload_keplerian_data(cursor, keplerian_data):
    cursor.execute("DELETE FROM keplerian")  # Delete all old records
    insert_keplerian = "INSERT INTO keplerian(satcat, sma, ecc, inc, raan, om, stamp) VALUES(%s, %s, %s, %s, %s, %s, %s)"
    cursor.executemany(insert_keplerian, keplerian_data)

def get_tle_data(cursor):
    # Data should be list of tuple(satcat: str, tle: str)
    select_tle = "SELECT * FROM tle"
    cursor.execute(select_tle)
    return cursor.fetchall()

def get_latest_tle_upload_timestamp(cursor):
    select_timestamp = "SELECT * FROM tle_upload ORDER BY stamp DESC LIMIT 1"
    cursor.execute(select_timestamp)
    return cursor.fetchall()

def get_propagation_data(cursor):
    select_propagation = "SELECT * FROM propagation"
    cursor.execute(select_propagation)
    return cursor.fetchall()

def get_keplerian_data(cursor, limit=1000):
    select_keplerian = f"SELECT satcat, sma, ecc, inc, raan, om FROM keplerian ORDER BY stamp DESC LIMIT {limit}"
    cursor.execute(select_keplerian)
    return cursor.fetchall()

def delete_conjunction_data(cursor):
    cursor.execute("DELETE FROM conjunctions")  # Delete all old records

def upload_conjunction(cursor, conjunction_data):
    # Data should be list of tuple(satcat: str, apogee_km: float, perigee_km: float, data: str (serialized JSON))
    insert_conjunctions = "INSERT INTO conjunctions(satcat1, satcat2, data) VALUES(%s, %s, %s)"
    cursor.executemany(insert_conjunctions, conjunction_data)

def get_conjunctions_data(cursor):
    select_conjunctions = "SELECT * FROM conjunctions"
    cursor.execute(select_conjunctions)
    return cursor.fetchall()