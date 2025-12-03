import os
import numpy as np
import logging
from datetime import datetime
from mangroves.geometry import Region
from mangroves.collection import Collection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Embeddings:

    def __init__(self) -> None:
        self.latitude_deg = None
        self.longitude_deg = None
        self.year = None
        self.regionDiameter_p = None
        self.spatialResolution_m = None
        self.region = None
        self.data = None
        self.band_names = None

    def _verify_year(self) -> bool:
        """
        Verify if the year is within valid range for AlphaEarth data.
        """
        if self.year < 2017 or self.year > datetime.now().year:
            logger.error(f'Invalid year: {self.year}. Must be between 2017 and {datetime.now().year}.')
            return False
        return True
    
    def from_collection(
            self, 
            latitude_deg: float, 
            longitude_deg: float, 
            year: int, 
            regionDiameter_p: int, 
            spatialResolution_m: int, 
            collection: Collection) -> None:
        """
        Extract the embedding patch for the specified region and year.

        Args:
            collection (Collection): The Collection instance to fetch data from.
        Returns:
            None
        """
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.year = year
        self.regionDiameter_p = regionDiameter_p
        self.spatialResolution_m = spatialResolution_m
        
        self.region = Region(
            latitude_deg, 
            longitude_deg, 
            regionDiameter_p
        )

        self.data = collection.extract(self.region, self.year)

    def from_file(
            self, 
            input_path: str) -> None:
        try:
            with np.load(input_path) as npzfile:
                self.latitude_deg = npzfile['latitude_deg']
                self.longitude_deg = npzfile['longitude_deg']
                self.year = npzfile['year']
                self.regionDiameter_p = npzfile['regionDiameter_p']
                self.spatialResolution_m = npzfile['spatialResolution_m']
                self.region = Region(
                    self.latitude_deg, 
                    self.longitude_deg, 
                    self.regionDiameter_p, 
                    self.spatialResolution_m
                )
                self.data = npzfile['data']
        except Exception as e:
            logger.error(f'Error loading patch from {input_path}: {e}')
    
    def save(
            self, 
            output_path: str, 
            feature_id: int) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create band names array (matching original format)
            band_names = [f'A{i:02d}' for i in range(64)]
            
            np.savez_compressed(
                output_path,
                data=self.data,
                feature_id=feature_id,
                longitude_deg=self.longitude_deg,
                latitude_deg=self.latitude_deg,
                regionDiameter_p=self.regionDiameter_p,
                spatialResolution_m=self.spatialResolution_m,
                year=self.year,
                num_images=1,
                band_names=band_names,
                flipud_applied=True
            )
            return True
        except Exception as e:
            logger.error(f'Error saving patch to {output_path}: {e}')
            return False
        