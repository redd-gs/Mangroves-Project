import math
import numpy as np
import ee
import logging

from mangroves.constants import RADIUS_EARTH_M, SPATIAL_RESOLUTION_M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Region:

    def __init__(
            self, 
            lat_deg: float, 
            lon_deg: float, 
            nPixels: int) -> None:
        self.lat0_deg = lat_deg
        self.lon0_deg = lon_deg
        self.nPixels = nPixels
        self.pts = []
        self.coords = {}

        assert self._verify_coordinates()
        self.region = self._get_region()

    def _verify_coordinates(self) -> bool:
        """
        Verify if the coordinates are within valid ranges.
        """
        if not (-90 <= self.lat0_deg <= 90):
            logger.error(f'Invalid latitude: {self.lat0_deg}. Must be between -90 and 90.')
            return False
        if not (-180 <= self.lon0_deg <= 180):
            logger.error(f'Invalid longitude: {self.lon0_deg}. Must be between -180 and 180.')
            return False
        return True
    
    def _get_region(self) -> ee.Geometry.Rectangle:
        distance_m = self.nPixels / 2 * SPATIAL_RESOLUTION_M

        # thetas_deg = [45, 135, 225, 315]
        thetas_deg = [0, 90, 180, 270]  # BBox are defined with North, East, South, West

        lat0_rad = math.radians(self.lat0_deg)
        lon0_rad = math.radians(self.lon0_deg)
        delta = distance_m / RADIUS_EARTH_M
        self.pts = []
        for theta_deg in thetas_deg:
            theta_rad = math.radians(theta_deg)  
            sin_lat = math.sin(lat0_rad) * math.cos(delta) + math.cos(lat0_rad) * math.sin(delta) * math.cos(theta_rad)
            lat_rad = math.asin(sin_lat)
            y = math.sin(theta_rad) * math.sin(delta) * math.cos(lat0_rad)
            x = math.cos(delta) - math.sin(lat0_rad) * sin_lat
            lon_rad = lon0_rad + math.atan2(y, x)
            lon_rad = (lon_rad + math.pi) % (2 * math.pi) - math.pi  # Normalize lon to [-pi, pi]
            lat_deg = math.degrees(lat_rad)
            lon_deg = math.degrees(lon_rad)
            self.pts.append((lat_deg, lon_deg))

        xMin, xMax, yMin, yMax = (
            min(lon for _, lon in self.pts), max(lon for _, lon in self.pts),
            min(lat for lat, _ in self.pts), max(lat for lat, _ in self.pts)
        )
        self.coords = {'xMin': xMin, 'xMax': xMax, 'yMin': yMin, 'yMax': yMax}

        region = ee.Geometry.Rectangle([xMin, yMin, xMax, yMax])
        return region




    # def _get_region(self) -> ee.Geometry.Rectangle:
    #     """
    #     Get a rectangular region around a point given size in pixels and spatial resolution.

    #     Args:
    #         latitude_deg (float): Latitude of the center point in degrees.
    #         longitude_deg (float): Longitude of the center point in degrees.
    #         regionDiameter_p (int): Diameter of the region in pixels.
    #         spatialResolution_m (int): Spatial resolution in meters per pixel.
    #     Returns:
    #         ee.Geometry.Rectangle: The rectangular region.
    #     """
    #     regionRadius_m = (self.regionDiameter_p - 1) * self.spatialResolution_m / 2  
    #     latitude_rad = np.radians(self.latitude_deg)
    #     meters_per_deg_lat = 111320
    #     meters_per_deg_lon = 111320 * np.cos(latitude_rad)
        
    #     latitudeHalfSize_deg = regionRadius_m / meters_per_deg_lat
    #     longitudeHalfSize_deg = regionRadius_m / meters_per_deg_lon
        
    #     self.west = self.longitude_deg - longitudeHalfSize_deg
    #     self.east = self.longitude_deg + longitudeHalfSize_deg
    #     self.south = self.latitude_deg - latitudeHalfSize_deg
    #     self.north = self.latitude_deg + latitudeHalfSize_deg
        
    #     region = ee.Geometry.Rectangle([self.west, self.south, self.east, self.north])
    #     return region
    