# Data Format Templates

Example file formats to help you prepare custom input data.

## sample_locations.csv

Example format for custom location coordinates:

```csv
latitude,longitude,year,date
1.2345,103.6789,2020,2020-01-15
-5.4321,150.1234,2020,2020-03-22
```

Use this format if you want to download embeddings for custom locations.

## Notes

- All coordinates must be in EPSG:4326 (WGS84)
- Year must be between 2017 and present (AlphaEarth availability)
- Date format: YYYY-MM-DD
