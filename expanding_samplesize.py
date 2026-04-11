import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta

# ============================================================
# LOAD DATA
# ============================================================

water = pd.read_csv("water_quality_allinfo_master.csv")
gdb_path = "C:\\Users\\Val\\CS2500\\DSFinalProject\\fire24_1.gdb"  # update this

fires = gpd.read_file(gdb_path, engine="pyogrio", layer="firep24_1")
fires = fires.to_crs("EPSG:4326")

# ============================================================
# CONVERT WATER TO GEODATAFRAME
# ============================================================

water['SampleDate'] = pd.to_datetime(water['SampleDate'])

water_gdf = gpd.GeoDataFrame(
    water,
    geometry=gpd.points_from_xy(water['TargetLongitude'], water['TargetLatitude']),
    crs="EPSG:4326"
)

# ============================================================
# PREPARE FIRE DATA
# ============================================================

fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'], utc=True).dt.tz_localize(None)
fires['CONT_DATE'] = pd.to_datetime(fires['CONT_DATE'], utc=True).dt.tz_localize(None)

# define the window: fire start to 1 month after fire start
fires['window_start'] = fires['ALARM_DATE']
fires['window_end'] = fires['ALARM_DATE'] + timedelta(days=30)

# ============================================================
# SPATIAL JOIN — stations inside fire perimeters
# ============================================================

print("Running spatial join...")
spatially_joined = gpd.sjoin(
    water_gdf,
    fires[['FIRE_NAME', 'ALARM_DATE', 'window_start', 'window_end', 'GIS_ACRES', 'geometry']],
    how='inner',       # only keep stations that fall inside a fire perimeter
    predicate='within'
)

print(f"Stations spatially inside a fire perimeter: {len(spatially_joined)}")

# ============================================================
# TEMPORAL FILTER — sample taken within 1 month after fire
# ============================================================

spatially_joined['SampleDate'] = pd.to_datetime(spatially_joined['SampleDate'])

impacted_1month = spatially_joined[
    (spatially_joined['SampleDate'] >= spatially_joined['window_start']) &
    (spatially_joined['SampleDate'] <= spatially_joined['window_end'])
]

print(f"Stations inside fire AND sampled within 1 month after fire: {len(impacted_1month)}")
print(f"\nFire names in impacted set:\n{impacted_1month['FIRE_NAME'].value_counts()}")

# ============================================================
# SAVE RESULT
# ============================================================

impacted_1month.to_csv("impacted_within_1month.csv", index=False)
print("\nSaved to impacted_within_1month.csv")
print(f"\nSample date range: {impacted_1month['SampleDate'].min()} to {impacted_1month['SampleDate'].max()}")
print(f"\nPreview:")
print(impacted_1month[['StationName', 'SampleDate', 'FIRE_NAME', 'GIS_ACRES', 'Oxygen, Dissolved, Total']].head(10))