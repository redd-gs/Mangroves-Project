import numpy as np
import ee
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Region:

    def __init__(
            self, 
            latitude_deg: float, 
            longitude_deg: float, 
            regionDiameter_p: int, 
            spatialResolution_m: int) -> None:
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.centering = False
        if regionDiameter_p % 2 == 0:
            regionDiameter_p += 1  # Ensure odd diameter for centering
            self.centering = True
        self.regionDiameter_p = regionDiameter_p
        self.spatialResolution_m = spatialResolution_m
        self.area_m2 = (regionDiameter_p * spatialResolution_m) ** 2

        assert self._verify_coordinates()
        self.region = self._get_region()

    def _verify_coordinates(self) -> bool:
        """
        Verify if the coordinates are within valid ranges.
        """
        if not (-90 <= self.latitude_deg <= 90):
            logger.error(f'Invalid latitude: {self.latitude_deg}. Must be between -90 and 90.')
            return False
        if not (-180 <= self.longitude_deg <= 180):
            logger.error(f'Invalid longitude: {self.longitude_deg}. Must be between -180 and 180.')
            return False
        return True

    def _get_region(self) -> ee.Geometry.Rectangle:
        """
        Get a rectangular region around a point given size in pixels and spatial resolution.

        Args:
            latitude_deg (float): Latitude of the center point in degrees.
            longitude_deg (float): Longitude of the center point in degrees.
            regionDiameter_p (int): Diameter of the region in pixels.
            spatialResolution_m (int): Spatial resolution in meters per pixel.
        Returns:
            ee.Geometry.Rectangle: The rectangular region.
        """
        regionRadius_m = (self.regionDiameter_p - 1) * self.spatialResolution_m / 2  
        latitude_rad = np.radians(self.latitude_deg)
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(latitude_rad)
        
        latitudeHalfSize_deg = regionRadius_m / meters_per_deg_lat
        longitudeHalfSize_deg = regionRadius_m / meters_per_deg_lon
        
        self.west = self.longitude_deg - longitudeHalfSize_deg
        self.east = self.longitude_deg + longitudeHalfSize_deg
        self.south = self.latitude_deg - latitudeHalfSize_deg
        self.north = self.latitude_deg + latitudeHalfSize_deg
        
        region = ee.Geometry.Rectangle([self.west, self.south, self.east, self.north])
        return region
    