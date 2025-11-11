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
        pass

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
            np.ndarray or None: The extracted embedding patch, or None if extraction fails.
        """
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.year = year
        self.regionDiameter_p = regionDiameter_p
        self.spatialResolution_m = spatialResolution_m
        
        self.region = Region(
            latitude_deg, 
            longitude_deg, 
            regionDiameter_p, 
            spatialResolution_m
        )

        self.data = collection.extract(self.region, self.year)

    def from_file(
            self, 
            input_path: str) -> bool:
        try:
            with np.load(input_path) as npzfile:
                self.data = npzfile['image_data']
            return True
        except Exception as e:
            logger.error(f'Error loading patch from {input_path}: {e}')
            return False
    
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
                image_data=self.data,
                feature_id=feature_id,
                centroid_lon=self.longitude_deg,
                centroid_lat=self.latitude_deg,
                year=self.year,
                num_images=1,
                band_names=band_names,
                flipud_applied=True
            )
            return True
        except Exception as e:
            logger.error(f'Error saving patch to {output_path}: {e}')
            return False
        