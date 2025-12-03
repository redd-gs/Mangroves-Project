RADIUS_EARTH_M = 6371000.

SPATIAL_RESOLUTION_M = 10.

REGION_DIAMETER_P = 244.    # 244x244 pixels

N_BANDS = 64

BANDS = [f'A{i:02d}' for i in range(N_BANDS)]

S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B', 'MSK_CLDPRB', 'MSK_SNWPRB', 'QA10', 'QA20', 'QA60', 'MSK_CLASSI_OPAQUE', 'MSK_CLASSI_CIRRUS', 'MSK_CLASSI_SNOW_ICE']

RGB_BANDS = ['B4', 'B3', 'B2']  # Red, Green, Blue
