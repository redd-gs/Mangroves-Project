# Raw External Data

Place your external data sources here.

## Required Files

### 1. GMW v3 2020 Shapefile

**Download from:** https://data.unep-wcmc.org/datasets/45
**Place in:** `gmw_v3_2020/`

Required files:
- gmw_v3_2020_vec.shp
- gmw_v3_2020_vec.shx
- gmw_v3_2020_vec.dbf
- gmw_v3_2020_vec.prj

**Description:**
Global Mangrove Watch v3.0 (2020) vector dataset with 1,076,117 polygon features in EPSG:4326.

### 2. Google Earth Engine Credentials

**Obtain from:** https://code.earthengine.google.com/
**Place in:** `gee_credentials/gee_project_key.txt`

**Format:** Simple text file with your GEE project ID:
```
your-project-id-here
```

⚠️ **Security:** Never commit this file to git (protected by .gitignore)
