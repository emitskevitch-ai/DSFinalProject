# PURPOSE:
#   This is the data preparation code. It does two things:
#
#   1. CONSOLIDATE RAW DATA — Reads all individual analyte CSV files from the
#      Analytes/ folder and merges them into a single master CSV
#      where each row is one water station reading and each column is an analyte.
#
#   2. SPATIAL JOIN — Takes the master water quality data and overlays it with
#      California wildfire perimeters using geographic coordinates. This tells
#      us which water stations were physically located inside a fire perimeter,
#      and during what time period.
#
# INPUTS:
#   - Analytes/*.csv                     — Raw water quality data, one file per analyte
#   - fire24_1.gdb                       — California fire perimeters geodatabase
#
# OUTPUTS (all sent to csvs/):
#   - water_quality_allinfo_master.csv   — Master wide-format water quality table
#   - analyte_units_reference.csv        — Reference table of units per analyte
#   - stations_within_fires_sameyear.csv — Stations inside a fire perimeter in
#                                          the same year as the fire
#   - all_stations_fire_joined.csv       — All stations with fire info attached
#                                          (NaN columns where no fire match)

import os
import glob
import pandas as pd
import geopandas as gpd

# os import allows us to easily navigate to files withn folders
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_CSVS = os.path.join(_ROOT, "csvs")

# STEP 1: MASTER WATER QUALITY TABLE
def build_master_water_quality(files):
    """
    Reads all individual analyte CSV files, stacks them into one long-format
    table, then pivots to wide format so each analyte becomes its own column.
    Parameters:
        files (list): List of file paths to analyte CSV files.
    Outputs:
        Writes water_quality_allinfo_master.csv and analyte_units_reference.csv
        to the csvs/ folder.
    """
    # Load all analyte files and stack them vertically
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    # Sort so that readings from the same station are grouped together chronologically
    combined = combined.sort_values(['StationCode', 'SampleDate']).reset_index(drop=True)
    print(f"Analytes found: {list(combined['Analyte'].unique())}")
    # Save a units reference table before pivoting 
    units_ref = combined.groupby('Analyte')['Unit'].first().reset_index()
    units_ref.to_csv(os.path.join(_CSVS, "analyte_units_reference.csv"), index=False)
    # Pull out station data before pivoting
    metadata = combined[['StationCode', 'SampleDate', 'StationName', 'TargetLatitude', 'TargetLongitude', 'Region']].drop_duplicates()
    # Pivot to wide format
    master = combined.pivot_table(index=['StationCode', 'SampleDate'], columns='Analyte', values='Result', aggfunc='first').reset_index()
    # Remove the column axis name that pivot_table adds automatically
    master.columns.name = None
    # --- Re-attach station metadata ---
    master = pd.merge(master, metadata, on=['StationCode', 'SampleDate'], how='left')
    # --- Add a unique row ID as the first column ---
    master.insert(0, 'WaterQualityID', range(1, len(master) + 1))
    # --- Reorder columns: metadata first, then analytes ---
    meta_cols = ['WaterQualityID', 'StationCode', 'StationName','TargetLatitude', 'TargetLongitude', 'Region', 'SampleDate']
    analyte_cols = [col for col in master.columns if col not in meta_cols]
    master = master[meta_cols + analyte_cols]
    # Final sort by station then date
    master = master.sort_values(['StationCode', 'SampleDate']).reset_index(drop=True)
    # --- Export ---
    master.to_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"), index=False)
    print(f"Saved water_quality_allinfo_master.csv  ({master.shape[0]} rows, {master.shape[1]} columns)")
    print(master.head())

# STEP 2: SPATIAL JOIN — MATCH WATER STATIONS TO FIRE PERIMETERS
def build_fire_spatial_join(wq):
    """
    Overlays water quality station locations (lat/lon points) on top of
    California wildfire perimeters (polygons) to find which stations were
    physically located inside a fire boundary.
    Parameters:
        wq (DataFrame): The master water quality table with lat/lon columns.
    Outputs:
        Writes stations_within_fires_sameyear.csv and all_stations_fire_joined.csv
        to the csvs/ folder.
    """
    # --- Parse dates and extract the year of each water sample ---
    wq['SampleDate'] = pd.to_datetime(wq['SampleDate'])
    wq['SampleYear'] = wq['SampleDate'].dt.year
    # --- Convert water quality data to a GeoDataFrame ---
    water_gdf = gpd.GeoDataFrame(wq, geometry=gpd.points_from_xy(wq['TargetLongitude'], wq['TargetLatitude']), crs="EPSG:4326")
    # --- Load California wildfire perimeters from the .gdb file ---
    fires = gpd.read_file(os.path.join(_ROOT, "fire24_1.gdb"), engine="pyogrio", layer="firep24_1")
    fires = fires.to_crs("EPSG:4326")
    # Parse the fire alarm date and extract the fire year
    fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'])
    fires['FireYear'] = fires['ALARM_DATE'].dt.year
    # --- Spatial join: find every station that falls inside a fire perimeter ---
    joined = gpd.sjoin(water_gdf, fires[['FIRE_NAME', 'ALARM_DATE', 'FireYear', 'GIS_ACRES', 'geometry']], how='left', predicate='within')
    # --- Filter to same-year matches ---
    same_year = joined[joined['FIRE_NAME'].notna() & (joined['SampleYear'] == joined['FireYear'])]
    print(f"Total water quality readings: {len(water_gdf)}")
    print(f"Readings inside a fire perimeter (same year): {len(same_year)}")
    print(same_year[['StationCode', 'StationName', 'SampleDate', 'FIRE_NAME', 'GIS_ACRES']].head())
    # --- Export both versions ---
    # 1. Same-year impacted stations only (most useful for analysis)
    same_year.drop(columns='geometry').to_csv(os.path.join(_CSVS, "stations_within_fires_sameyear.csv"), index=False)
    print("Saved stations_within_fires_sameyear.csv")
    # 2. All stations with fire info attached (NaN where no fire match)
    joined.drop(columns='geometry').to_csv(os.path.join(_CSVS, "all_stations_fire_joined.csv"), index=False)

# MAIN — runs both steps in order
def main():
    """
    Runs the full data pipeline:
        1. Build master water quality CSV from raw analyte files.
        2. Spatially join water stations with fire perimeters.
    """
    # --- Step 1: Build master water quality table ---
    analyte_files = glob.glob(os.path.join(_ROOT, "Analytes_reduced", "*.csv"))
    if not analyte_files:
        print("ERROR: No CSV files found in Analytes/ folder. Check the path.")
        return
    print(f"Found {len(analyte_files)} analyte files.")
    build_master_water_quality(analyte_files)
    # --- Step 2: Spatial join ---
    wq = pd.read_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"))
    build_fire_spatial_join(wq)
if __name__ == "__main__":
    main()
