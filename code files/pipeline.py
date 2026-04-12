# =============================================================================
# pipeline.py
# =============================================================================
# PURPOSE:
#   This is the data preparation script. It must be run FIRST before any
#   analysis scripts. It does two things:
#
#   1. CONSOLIDATE RAW DATA — Reads all individual analyte CSV files from the
#      Analytes/ folder and merges them into a single wide-format master CSV
#      where each row is one water station reading and each column is an analyte.
#
#   2. SPATIAL JOIN — Takes the master water quality data and overlays it with
#      California wildfire perimeters using geographic coordinates. This tells
#      us which water stations were physically located inside a fire perimeter,
#      and during what time period.
#
# INPUTS:
#   - Analytes/*.csv         — Raw water quality data, one file per analyte
#   - fire24_1.gdb           — California fire perimeters geodatabase (2024)
#
# OUTPUTS (all written to csvs/):
#   - water_quality_allinfo_master.csv   — Master wide-format water quality table
#   - analyte_units_reference.csv        — Reference table of units per analyte
#   - stations_within_fires_sameyear.csv — Stations inside a fire perimeter in
#                                          the same calendar year as the fire
#   - all_stations_fire_joined.csv       — All stations with fire info attached
#                                          (NaN columns where no fire match)
# =============================================================================

import os
import glob
import pandas as pd
import geopandas as gpd

# -----------------------------------------------------------------------------
# PATH SETUP
# This script lives in "code files/" so we navigate one level up (..) to reach
# the project root where fire24_1.gdb and the Analytes/ folder live.
# All output CSVs go into the csvs/ folder.
# -----------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))   # .../code files/
_ROOT  = os.path.join(_HERE, "..")                     # project root
_CSVS  = os.path.join(_ROOT, "csvs")                  # output folder for CSVs


# =============================================================================
# STEP 1: BUILD THE MASTER WATER QUALITY TABLE
# =============================================================================
def build_master_water_quality(files):
    """
    Reads all individual analyte CSV files, stacks them into one long-format
    table, then pivots to wide format so each analyte becomes its own column.

    Why pivot? The raw files store data in "long" format — each row is one
    measurement of one analyte. We want "wide" format — each row is one
    station/date combination with all analytes as separate columns. This makes
    it much easier to compare analytes side-by-side and run multi-variable
    analyses.

    Parameters:
        files (list): List of file paths to analyte CSV files.

    Outputs:
        Writes water_quality_allinfo_master.csv and analyte_units_reference.csv
        to the csvs/ folder.
    """

    print("=" * 60)
    print("STEP 1: Building master water quality table")
    print("=" * 60)

    # --- Load all analyte files and stack them vertically ---
    # Each file has the same columns but covers a different chemical analyte.
    # pd.concat stacks them into one tall DataFrame.
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Sort so that readings from the same station are grouped together
    # chronologically — makes the data easier to inspect manually.
    combined = combined.sort_values(['StationCode', 'SampleDate']).reset_index(drop=True)

    print(f"  Analytes found: {list(combined['Analyte'].unique())}")

    # --- Save a units reference table before pivoting ---
    # Once we pivot, the unit information is lost from the main table.
    # We save it separately so we can always look up what unit each analyte uses.
    units_ref = combined.groupby('Analyte')['Unit'].first().reset_index()
    units_ref.to_csv(os.path.join(_CSVS, "analyte_units_reference.csv"), index=False)
    print("  Saved analyte_units_reference.csv")

    # --- Pull out station metadata before pivoting ---
    # The pivot operation only keeps the columns we pivot on (StationCode,
    # SampleDate) and the values column (Result). Station metadata like name,
    # lat/lon, and region would be lost, so we extract it first and re-join
    # it after the pivot.
    metadata = combined[
        ['StationCode', 'SampleDate', 'StationName',
         'TargetLatitude', 'TargetLongitude', 'Region']
    ].drop_duplicates()

    # --- Pivot to wide format ---
    # Before: one row per (station, date, analyte)
    # After:  one row per (station, date), one column per analyte
    # aggfunc='first' handles any duplicate readings for the same
    # station/date/analyte by just keeping the first one.
    master = combined.pivot_table(
        index=['StationCode', 'SampleDate'],
        columns='Analyte',
        values='Result',
        aggfunc='first'
    ).reset_index()

    # Remove the column axis name that pivot_table adds automatically
    master.columns.name = None

    # --- Re-attach station metadata ---
    master = pd.merge(master, metadata, on=['StationCode', 'SampleDate'], how='left')

    # --- Add a unique row ID as the first column ---
    master.insert(0, 'WaterQualityID', range(1, len(master) + 1))

    # --- Reorder columns: metadata first, then analytes ---
    # This makes the table more readable — you always see the station info
    # before the measurement values.
    meta_cols = ['WaterQualityID', 'StationCode', 'StationName',
                 'TargetLatitude', 'TargetLongitude', 'Region', 'SampleDate']
    analyte_cols = [col for col in master.columns if col not in meta_cols]
    master = master[meta_cols + analyte_cols]

    # Final sort by station then date
    master = master.sort_values(['StationCode', 'SampleDate']).reset_index(drop=True)

    # --- Export ---
    master.to_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"), index=False)
    print(f"  Saved water_quality_allinfo_master.csv  ({master.shape[0]} rows, {master.shape[1]} columns)")
    print(master.head())


# =============================================================================
# STEP 2: SPATIAL JOIN — MATCH WATER STATIONS TO FIRE PERIMETERS
# =============================================================================
def build_fire_spatial_join(wq):
    """
    Overlays water quality station locations (lat/lon points) on top of
    California wildfire perimeters (polygons) to find which stations were
    physically located inside a fire boundary.

    This uses GeoPandas for the spatial operations:
      - Water stations are converted to Point geometries using their lat/lon.
      - Fire perimeters are loaded as Polygon geometries from the .gdb file.
      - A spatial join finds every station/fire combination where the station
        falls inside the fire polygon.

    We also extract the fire's alarm year and filter to only keep station
    readings from the same calendar year as the fire — this gives us the most
    directly fire-relevant subset of readings.

    Parameters:
        wq (DataFrame): The master water quality table with lat/lon columns.

    Outputs:
        Writes stations_within_fires_sameyear.csv and all_stations_fire_joined.csv
        to the csvs/ folder.
    """

    print("\n" + "=" * 60)
    print("STEP 2: Spatial join — water stations vs fire perimeters")
    print("=" * 60)

    # --- Parse dates and extract the year of each water sample ---
    # We need the year so we can later filter to same-year fire/sample matches.
    wq['SampleDate'] = pd.to_datetime(wq['SampleDate'])
    wq['SampleYear'] = wq['SampleDate'].dt.year

    # --- Convert water quality data to a GeoDataFrame ---
    # GeoPandas needs a "geometry" column of Point objects to do spatial operations.
    # gpd.points_from_xy() creates those points from the existing lat/lon columns.
    # EPSG:4326 is standard WGS84 lat/lon — the coordinate system GPS uses.
    water_gdf = gpd.GeoDataFrame(
        wq,
        geometry=gpd.points_from_xy(wq['TargetLongitude'], wq['TargetLatitude']),
        crs="EPSG:4326"
    )

    # --- Load California wildfire perimeters from the .gdb file ---
    # fire24_1.gdb is an ESRI geodatabase containing polygon shapes of each fire.
    # "firep24_1" is the specific layer inside the geodatabase we want.
    # We reproject to EPSG:4326 to match the water station coordinate system.
    print("  Loading fire perimeters from fire24_1.gdb ...")
    fires = gpd.read_file(
        os.path.join(_ROOT, "fire24_1.gdb"),
        engine="pyogrio",
        layer="firep24_1"
    )
    fires = fires.to_crs("EPSG:4326")

    # Parse the fire alarm date and extract the fire year
    fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'])
    fires['FireYear'] = fires['ALARM_DATE'].dt.year

    # --- Spatial join: find every station that falls inside a fire perimeter ---
    # how='left' keeps ALL water station readings, adding fire columns as NaN
    # for stations that don't fall inside any fire perimeter.
    # predicate='within' means the station point must be inside the fire polygon.
    print("  Running spatial join ...")
    joined = gpd.sjoin(
        water_gdf,
        fires[['FIRE_NAME', 'ALARM_DATE', 'FireYear', 'GIS_ACRES', 'geometry']],
        how='left',
        predicate='within'
    )

    # --- Filter to same-year matches ---
    # A water sample taken in 2020 inside a 2018 fire boundary is probably not
    # fire-related. We keep only readings where the sample year matches the fire year.
    same_year = joined[
        joined['FIRE_NAME'].notna() &
        (joined['SampleYear'] == joined['FireYear'])
    ]

    print(f"  Total water quality readings: {len(water_gdf)}")
    print(f"  Readings inside a fire perimeter (same year): {len(same_year)}")
    print(same_year[['StationCode', 'StationName', 'SampleDate', 'FIRE_NAME', 'GIS_ACRES']].head())

    # --- Export both versions ---
    # 1. Same-year impacted stations only (most useful for analysis)
    same_year.drop(columns='geometry').to_csv(
        os.path.join(_CSVS, "stations_within_fires_sameyear.csv"), index=False
    )
    print("  Saved stations_within_fires_sameyear.csv")

    # 2. All stations with fire info attached (NaN where no fire match)
    # This is useful for baseline comparisons — you can see impacted vs
    # non-impacted stations in the same table.
    joined.drop(columns='geometry').to_csv(
        os.path.join(_CSVS, "all_stations_fire_joined.csv"), index=False
    )
    print("  Saved all_stations_fire_joined.csv")


# =============================================================================
# MAIN — runs both steps in order
# =============================================================================
def main():
    """
    Runs the full data pipeline:
      1. Build master water quality CSV from raw analyte files.
      2. Spatially join water stations with fire perimeters.

    Step 1 must complete before Step 2 because Step 2 reads the CSV that
    Step 1 produces.
    """

    # --- Step 1: Build master water quality table ---
    # Find all analyte CSV files in the Analytes/ folder
    analyte_files = glob.glob(os.path.join(_ROOT, "Analytes", "*.csv"))
    if not analyte_files:
        print("ERROR: No CSV files found in Analytes/ folder. Check the path.")
        return
    print(f"Found {len(analyte_files)} analyte files.")
    build_master_water_quality(analyte_files)

    # --- Step 2: Spatial join ---
    # Read the master CSV that Step 1 just wrote
    wq = pd.read_csv(os.path.join(_CSVS, "water_quality_allinfo_master.csv"))
    build_fire_spatial_join(wq)

    print("\n" + "=" * 60)
    print("Pipeline complete. CSVs written to csvs/")
    print("You can now run temporal_analysis.py and models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
