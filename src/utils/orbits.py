import numpy as np

def keplerian2ecef(a, e, inc, raan, om, n=100):
    true_anomaly = np.linspace(0, 2 * np.pi, n)
    r = a * (1 - e ** 2) / (1 + e * np.cos(true_anomaly))
    x_eci = r * (np.cos(om + true_anomaly)) 
    y_eci = r * (np.sin(om + true_anomaly))
    z_eci = np.zeros(n)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(inc), -np.sin(inc)],
                [0, np.sin(inc), np.cos(inc)]])
    Rz_raan = np.array([[np.cos(raan), -np.sin(raan), 0],
                    [np.sin(raan), np.cos(raan), 0],
                    [0, 0, 1]])
    # TODO: ECEF is wrong. Needs UT1 rotation.
    coords_ecef = []
    for i in range(n):
        pos_eci = np.array([x_eci[i], y_eci[i], z_eci[i]])
        pos_ecef = np.dot(Rz_raan, np.dot(Rx, pos_eci))
        coords_ecef.append(pos_ecef)
    coords_ecef = np.array(coords_ecef)
    x_ecef = coords_ecef[:, 0]
    y_ecef = coords_ecef[:, 1]
    z_ecef = coords_ecef[:, 2]
    return x_ecef, y_ecef, z_ecef

def ecef2latlon(x, y, z):
    """
    Converts ECEF coordinates to latitude, longitude, and altitude.

    Args:
        x (float): ECEF X-coordinate (meters).
        y (float): ECEF Y-coordinate (meters).
        z (float): ECEF Z-coordinate (meters).

    Returns:
        tuple: (latitude in degrees, longitude in degrees, altitude in meters)
    """
    a = 6378.137  # Semi-major axis of the Earth (WGS84)
    e2 = 0.00669438  # Eccentricity squared (WGS84)

    # Calculate longitude
    lon = np.arctan2(y, x)

    # Calculate distance from the Z-axis
    p = np.sqrt(x ** 2 + y ** 2)

    # Initial guess for latitude
    lat = np.arctan2(z, p * (1 - e2))

    # Iteratively improve latitude
    for i in range(10):  # Number of iterations
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    # Calculate altitude
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt

def keplerian2latlon(a, e, inc, raan, om, n=100):
    return ecef2latlon(*keplerian2ecef(a, e, inc, raan, om, n))

def random_keplerian():
    """Randomized Keplerian elements for fun."""
    a = np.random.randint(8500, 10000)
    e = np.random.rand() / 4
    inc = np.radians(np.random.randint(0, 90))
    raan = np.radians(np.random.randint(0, 90))
    om = np.radians(np.random.randint(0, 90))
    return a, e, inc, raan, om
