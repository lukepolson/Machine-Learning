import pandas as pd
import numpy as np


def reduce_mem_usage(df):
    result = df.copy()
    for col in result.columns:
        col_data = result[col]
        dn = col_data.dtype.name
        if dn == "object":
            result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")
        elif dn == "bool":
            result[col] = col_data.astype("int8")
        elif dn.startswith("int") or (col_data.round() == col_data).all():
            result[col] = pd.to_numeric(col_data, downcast="integer")
        else:
            result[col] = pd.to_numeric(col_data, downcast='float')
    return result

## READ WEATHER DATA
site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]

def read_weather_train(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    df = pd.read_csv(root + 'weather_train.csv', parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    if fix_timestamps:
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)
    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column
    elif add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()
    return reduce_mem_usage(df).set_index(["site_id", "timestamp"])

def read_weather_test(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    df = pd.read_csv(root + 'weather_test.csv', parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2017-01-01")).dt.total_seconds() // 3600
    if fix_timestamps:
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)
    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column
    elif add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()
    return reduce_mem_usage(df).set_index(["site_id", "timestamp"])

## READ BUILDING METADATA
def read_building_metadata():
    df = pd.read_csv(root+'building_metadata.csv')
    df['surf_area'] = np.sqrt(df['square_feet']/df['floor_count'])
    df = reduce_mem_usage(df.fillna(-1)).set_index("building_id")
    return df

## READ TRAINING AND TEST DATA
def read_train():
    df = pd.read_csv(root+'train.csv', parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    return reduce_mem_usage(df)

def read_test():
    df = pd.read_csv(root+'test.csv', parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2017-01-01")).dt.total_seconds() // 3600
    return reduce_mem_usage(df)


## COMBINE DATA
def combined_train_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    Xy = reduce_mem_usage(read_train().join(read_building_metadata(), on="building_id").join(
        read_weather_train(fix_timestamps, interpolate_na, add_na_indicators),
        on=["site_id", "timestamp"]).fillna(-1))
    return Xy #DataFrame containing labels and data (X and y)

def combined_test_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    Xy = reduce_mem_usage(read_test().join(read_building_metadata(), on="building_id").join(
        read_weather_test(fix_timestamps, interpolate_na, add_na_indicators),
        on=["site_id", "timestamp"]).fillna(-1))
    return Xy #DataFrame containing labels and data (X and y)

root = 'C:\\Users\\lukep\\Documents\\big_data\\ASHRAE\\'

# Also get properly formatted train and test data
Xy = combined_train_data()
Xy.to_pickle(root+'PROCESSED_TRAIN_DF.pkl')

Xy = combined_test_data()
Xy.to_pickle(root+'PROCESSED_TEST_DF.pkl')