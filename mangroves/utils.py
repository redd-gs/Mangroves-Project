import numpy as np
import math
from typing import Tuple, List
from mangroves.constants import RADIUS_EARTH_M 


def haversine(lat1_deg: float, lat2_deg: float, lon1_deg: float, lon2_deg: float, R: float = RADIUS_EARTH_M) -> float:
    """
    Calculate the Haversine distance between two geographic coordinates.
    """
    lat1_rad = np.radians(lat1_deg)
    lat2_rad = np.radians(lat2_deg)
    dlat_rad = np.radians(lat2_deg - lat1_deg)
    dlon_rad = np.radians(lon2_deg - lon1_deg)

    a = np.sin(dlat_rad / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def geodesic_circle(
        lat0_deg: float, 
        lon0_deg: float, 
        distance_m: float, 
        n_points: int = 360, 
        R: float = RADIUS_EARTH_M) -> List[Tuple[float, float]]:
    """
    Generate points approximating a circle around a reference point using geodesic calculations.
    """
    lat0_rad = math.radians(lat0_deg)
    lon0_rad = math.radians(lon0_deg)
    delta = distance_m / R
    pts = []
    for k in range(n_points):
        theta = 2 * math.pi * k / n_points  # bearing
        sin_lat = math.sin(lat0_rad) * math.cos(delta) + math.cos(lat0_rad) * math.sin(delta) * math.cos(theta)
        lat_rad = math.asin(sin_lat)
        y = math.sin(theta) * math.sin(delta) * math.cos(lat0_rad)
        x = math.cos(delta) - math.sin(lat0_rad) * sin_lat
        lon_rad = lon0_rad + math.atan2(y, x)
        lon_rad = (lon_rad + math.pi) % (2 * math.pi) - math.pi  # Normalize lon to [-pi, pi]
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)
        pts.append((lat_deg, lon_deg))
    return pts


def planar_approx_circle(
        lat0_deg: float, 
        lon0_deg: float, 
        distance_m: float, 
        n_points: int = 360, 
        R: float = RADIUS_EARTH_M) -> List[Tuple[float, float]]:
    """
    Generate points approximating a circle around a reference point using planar approximation.
    """
    K = R * (math.pi / 180)  # meters per degree latitude at equator (approx 111320 m)
    pts = []
    for k in range(n_points):
        theta = 2 * math.pi * k / n_points  # angle
        dlat_deg = (distance_m * math.cos(theta)) / K
        dlon_deg = (distance_m * math.sin(theta)) / (K * math.cos(lat0_deg * (math.pi / 180)))
        lat_deg = lat0_deg + dlat_deg
        lon_deg = lon0_deg + dlon_deg
        pts.append((lat_deg, lon_deg))
    return pts