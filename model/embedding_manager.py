import os
import numpy as np
import logging
from model.geo_utilities import Region
from model.gee_collection import Collection

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
        