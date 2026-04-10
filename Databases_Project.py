'''
gdf["longitude"] = gdf.geometry.x
gdf["latitude"] = gdf.geometry.y
df = gdf.drop(columns="geometry")
df.to_csv("output_with_coords.csv", index=False)
'''

# Eva Mitskevitch
# 4/5/26
# This is an attempt to join all of the analytes into a master spreadsheet
from functools import reduce
import pandas as pd
import glob
import geopandas as gpd
from shapely.geometry import Point


def makewaterqualitymaster(files):

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    # Stack all sheets on top of each other
    combined = pd.concat(dfs, ignore_index=True)

    # Sort by station first, date second
    combined = combined.sort_values(['StationCode', 'SampleDate']).reset_index(drop=True)

    # Check analyte names
    print(combined['Analyte'].unique())

    # Save units reference before pivoting
    units_ref = combined.groupby('Analyte')['Unit'].first().reset_index()
    units_ref.to_csv("analyte_units_reference.csv", index=False)

    # Pull metadata columns out before pivoting so we can join them back after
    metadata = combined[['StationCode', 'SampleDate', 'StationName', 
                        'TargetLatitude', 'TargetLongitude', 'Region']].drop_duplicates()

    # Pivot analytes into wide format
    master = combined.pivot_table(
        index=['StationCode', 'SampleDate'],
        columns='Analyte',
        values='Result',
        aggfunc='first'
    ).reset_index()

    # Clean up column naming
    master.columns.name = None

    # Join metadata back onto the pivoted master
    master = pd.merge(master, metadata, on=['StationCode', 'SampleDate'], how='left')

    # Add primary key as first column
    master.insert(0, 'WaterQualityID', range(1, len(master) + 1))

    # Define metadata column order
    meta_cols = ['WaterQualityID', 'StationCode', 'StationName', 
                'TargetLatitude', 'TargetLongitude', 'Region', 'SampleDate']

    # Get analyte columns (everything not in metadata)
    analyte_cols = [col for col in master.columns if col not in meta_cols]

    # Reorder: metadata first, analytes after
    master = master[meta_cols + analyte_cols]

    # Sort by station then date
    master = master.sort_values(['StationCode', 'SampleDate']).reset_index(drop=True)

    # Export
    master.to_csv("water_quality_allinfo_master.csv", index=False)
    print(master.head())
    print(master.shape)

def makefiregeo(wq):

    # Convert SampleDate to datetime and extract year
    wq['SampleDate'] = pd.to_datetime(wq['SampleDate'])
    wq['SampleYear'] = wq['SampleDate'].dt.year

    # Convert to GeoDataFrame using lat/lon columns
    water_gdf = gpd.GeoDataFrame(
        wq,
        geometry=gpd.points_from_xy(wq['TargetLongitude'], wq['TargetLatitude']),
        crs="EPSG:4326"
    )

    # --- Load wildfire perimeters ---
    fires = gpd.read_file("C:/Users/Val/CS2500/fire24_1.gdb", 
                        engine="pyogrio", layer="firep24_1")
    fires = fires.to_crs("EPSG:4326")

    # Extract fire year from alarm date
    fires['ALARM_DATE'] = pd.to_datetime(fires['ALARM_DATE'])
    fires['FireYear'] = fires['ALARM_DATE'].dt.year

    # --- Spatial join ---
    joined = gpd.sjoin(
        water_gdf,
        fires[['FIRE_NAME', 'ALARM_DATE', 'FireYear', 'GIS_ACRES', 'geometry']],
        how='left',
        predicate='within'
    )

    # --- Filter to only matches where the sample year == fire year ---
    same_year = joined[
        joined['FIRE_NAME'].notna() & 
        (joined['SampleYear'] == joined['FireYear'])]

    # --- Results ---
    print(f"Total water quality readings: {len(water_gdf)}")
    print(f"Readings within a fire perimeter in the same year: {len(same_year)}")
    print(same_year[['StationCode', 'StationName', 'SampleDate', 'FIRE_NAME', 'GIS_ACRES']].head())

    # --- Export ---
    same_year.drop(columns='geometry').to_csv("stations_within_fires_sameyear.csv", index=False)

    # Also export all stations with fire info attached (NaN where no fire match)
    joined.drop(columns='geometry').to_csv("all_stations_fire_joined.csv", index=False)

def main():

    # Load all CSVs
    files = glob.glob("C:/Users/Val/CS2500/DSFinalProject/Analytes/*.csv")
    wq = pd.read_csv("water_quality_allinfo_master.csv")

    makewaterqualitymaster(files)

    makefiregeo(wq)




if __name__ == "__main__":
    main()