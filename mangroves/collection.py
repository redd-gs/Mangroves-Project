import ee
import numpy as np
import logging
from mangroves.geometry import Region

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Collection:

    def __init__(
            self, 
            project: str) -> None:
        self.project = project
        self.band_names = [f'A{i:02d}' for i in range(64)]
        self._initialize_gee()

    def _initialize_gee(self) -> bool:
        """
        Initialize Google Earth Engine with the specified project.

        Args:
            project (str): The GEE project ID.
        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project)
            logger.info('Google Earth Engine initialized successfully with service account')
            return True
        except Exception as e:
            logger.error(f'Failed to initialize GEE: {e}')
            return False

    def fetch_image_from_region_in_collection(
            self, 
            region: Region, 
            year: int) -> ee.Image:
        """
        Fetches the first available image from the Google Earth Engine ImageCollection
        for the specified coordinates and year.

        Args:
            region (Region): The region to fetch the image from.
            year (int): The year for which to fetch the image.
        Returns:
            ee.Image or None: The first image if available, else None.
        """
        embedding_collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
        filtered_collection = embedding_collection.filterBounds(region.region).filterDate(
            f'{year}-01-01', f'{year+1}-01-01'
        )

        count = filtered_collection.size().getInfo()
        logger.info(f'Filtered images for {year}: {count}')
            
        if count > 0:
            return filtered_collection.first()
        else:
            return None
        
    def is_available(
            self, 
            region: Region, 
            year: int) -> bool:
        """
        Check if AlphaEarth embedding data is available for a given point and year.

        Args:
            region (Region): The region to check availability for.
            year (int): The year to check availability for.
        Returns:
            bool: True if data is available, False otherwise.
        """
        try:
            image = self.fetch_image_from_region_in_collection(region.region, year)
            
            if image is not None:
                image_info = image.getInfo()
                logger.info(f'Image properties: {list(image_info.keys())}')
                
                # Check band names
                if 'bands' in image_info:
                    band_names = [band['id'] for band in image_info['bands']]
                    logger.info(f'Available bands: {band_names[:10]}... (showing first 10)')
                
                return True
            else:
                logger.warning(f'No AlphaEarth data found for {year}')
                return False
                
        except Exception as e:
            logger.error(f'An error occured during the check: {e}')
            return False
        
    def extract(
            self, 
            region: Region,
            year: int) -> np.ndarray:
        """
        Extract the embedding patch for the specified region and year.

        Args:
            region (Region): The region to extract the patch from.
            year (int): The year for which to extract the patch.
        Returns:
            np.ndarray or None: The extracted patch, or None if extraction fails.
        """
        try:
            image = self.fetch_image_from_region_in_collection(region, year)
            if image is None:
                logger.warning('No image found for the specified region')
                return None
            
            # Sample the image using sampleRectangle with timeout
            pixel_data = image.sampleRectangle(
                region=region.region,
                defaultValue=0,
                properties=[]
            )
            
            # Get the values with timeout
            pixel_dict = pixel_data.getInfo()
            if not pixel_dict or 'properties' not in pixel_dict:
                logger.warning(f'No data found for point \
                               ({region.latitude_deg:.4f}, {region.longitude_deg:.4f}) in year {year}')
                return None
            
            # Extract embedding bands data
            properties = pixel_dict['properties']
            bands_data = {}
            target_size = int(region.nPixels)

            for i in range(64):
                band_name = f'A{i:02d}'
                if band_name in properties:
                    band_array = np.array(properties[band_name])
                    
                    # Crop or pad to target size
                    h, w = band_array.shape
                    
                    # Center crop if larger
                    if h > target_size:
                        start_h = (h - target_size) // 2
                        band_array = band_array[start_h:start_h+target_size, :]
                    if w > target_size:
                        start_w = (w - target_size) // 2
                        band_array = band_array[:, start_w:start_w+target_size]
                        
                    # Pad if smaller
                    h, w = band_array.shape
                    if h < target_size or w < target_size:
                        pad_h = max(0, target_size - h)
                        pad_w = max(0, target_size - w)
                        band_array = np.pad(band_array, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

                    # Apply flipud for correct display
                    band_array = np.flipud(band_array)
                    bands_data[band_name] = band_array
            
            if len(bands_data) == 0:
                logger.warning(f'No embedding bands found for point \
                               ({region.latitude_deg:.4f}, {region.longitude_deg:.4f}) in year {year}')
                return None
            
            logger.info(f'Successfully extracted {len(bands_data)} bands')
            
            # Stack all 64 bands into a 64×H×W array
            image_stack = []
            
            for band_name in self.band_names:
                if band_name in bands_data:
                    image_stack.append(bands_data[band_name])
                else:
                    # Fill missing bands with zeros
                    if bands_data:
                        image_shape = list(bands_data.values())[0].shape
                        image_stack.append(np.zeros(image_shape))
                    else:
                        return None
            
            # Stack to create D*H*W array
            patch = np.stack(image_stack, axis=0)
            
            logger.info(f'Successfully created patch with shape: {patch.shape}')
            return patch
        
        except Exception as e:
            logger.error(f'Error extracting patch for \
                         ({region.latitude_deg:.4f}, {region.longitude_deg:.4f}) in year {year}: {e}')
            return None
        