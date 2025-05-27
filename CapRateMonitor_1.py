# %% [markdown]
# # Data Analysis Script
# 
# This script is organized into cells for interactive analysis, similar to a Jupyter notebook.

# %% [markdown]
# ## Import Libraries

# %%
# import the major libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import geopandas as gpd
import pyarrow.parquet as pq
import plotly.express as px
import plotly as py
import folium
import json
from shapely.geometry import Point
import io
from pyproj import Proj, transform
import os
from scipy.stats import linregress

# Set rcParams to make sure text is saved as editable text in SVG files
plt.rcParams['svg.fonttype'] = 'none'  # Makes text in SVGs editable
plt.rcParams['font.sans-serif'] = 'Arial'  # (Optional) Set a default font like Arial

# set display to show max columns
pd.set_option('display.max_columns', None)

# set display to show 300 rows maximum
pd.set_option('display.max_rows', 50)

# Set display width to prevent line wrapping in tables
pd.set_option('display.width', 1000) # Adjust this value as needed for your screen width

# %% [markdown]
# ## Set Directory Paths

# %%
# Set main directory (current directory)
main_dir = os.getcwd()
print(f"Main directory: {main_dir}")

# Set reference directory
reference_dir = os.path.join(os.path.expanduser('~'), 'Home Economics Dropbox', 'Aziz Sunderji', 'Home Economics', 'Reference', 'Shapefiles')
print(f"Reference directory: {reference_dir}")

# %% # Load ZHVI data
print("# %% # Load ZHVI data")
price = pd.read_csv('../ZillowData/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv', encoding='latin-1')
price.head()

# %% # Load ZORI data
print("# %% # Load ZORI data")
rent = pd.read_csv('../ZillowData/Zip_zori_uc_sfrcondomfr_sm_month.csv', encoding='latin-1')
rent.head()

# %% # Inspect columns of price DataFrame
print("# %% # Inspect columns of price DataFrame")
print(price.columns)

# %% # Inspect columns of rent DataFrame
print("# %% # Inspect columns of rent DataFrame")
print(rent.columns)
# %%

# %% # Identify date columns and latest common date
print("# %% # Identify date columns and latest common date")
price_date_cols = [col for col in price.columns if isinstance(col, str) and col.count('-') == 2 and len(col) == 10] # Assuming YYYY-MM-DD format
rent_date_cols = [col for col in rent.columns if isinstance(col, str) and col.count('-') == 2 and len(col) == 10] # Assuming YYYY-MM-DD format

latest_price_date = max(price_date_cols)
latest_rent_date = max(rent_date_cols)

latest_common_date = min(latest_price_date, latest_rent_date)
print(f"Latest common date for price and rent data: {latest_common_date}")

# %% # Calculate Price-to-Rent Ratio for the latest month
print("# %% # Calculate Price-to-Rent Ratio for the latest month")
# Select relevant columns (RegionName, identifiers, and the latest date) from price dataframe
identifier_cols = ['RegionName', 'City', 'State', 'Metro', 'CountyName'] # Assuming these are the correct column names in your 'price' df
price_info_latest = price[identifier_cols + [latest_common_date]].rename(columns={latest_common_date: 'Price'})

# Select relevant columns (RegionName and the latest date) from rent dataframe
rent_values_latest = rent[['RegionName', latest_common_date]].rename(columns={latest_common_date: 'Rent'})

# Merge the two dataframes on RegionName
merged_df = pd.merge(price_info_latest, rent_values_latest, on='RegionName', how='inner')

# Calculate the Price-to-Rent Ratio
merged_df['Price_to_Rent_Ratio'] = merged_df['Price'] / merged_df['Rent']

# Display the first few rows of the merged dataframe with the ratio
print(merged_df.head())

# Display summary statistics for the ratio
print(merged_df['Price_to_Rent_Ratio'].describe())

# %% # Display Top and Bottom 10 Zip Codes by Price-to-Rent Ratio
print("# %% # Display Top and Bottom 10 Zip Codes by Price-to-Rent Ratio")
# Sort the DataFrame by Price-to-Rent Ratio
merged_df_sorted = merged_df.sort_values(by='Price_to_Rent_Ratio', ascending=False)

# Get the top 10
top_10 = merged_df_sorted.head(10)
print("Top 10 Zip Codes by Price-to-Rent Ratio:")
print(top_10)

# Get the bottom 10
bottom_10 = merged_df_sorted.tail(10)
print("\nBottom 10 Zip Codes by Price-to-Rent Ratio:")
print(bottom_10)

# %% # Visualize Price vs. Rent with a Trend Line
print("# %% # Visualize Price vs. Rent with a Trend Line")
# Remove rows with NaN or Inf in Price_to_Rent_Ratio if they exist, and also ensure Price and Rent are finite
merged_df_cleaned = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent', 'Price_to_Rent_Ratio'])

# Filter out rows where Rent is greater than $10,000
rent_cap = 10000
merged_df_cleaned = merged_df_cleaned[merged_df_cleaned['Rent'] <= rent_cap]

# Create a scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=merged_df_cleaned, x='Rent', y='Price', alpha=0.5)
sns.regplot(data=merged_df_cleaned, x='Rent', y='Price', scatter=False, color='red') # Adds a regression line

plt.title(f'Price vs. Rent for All Zip Codes ({latest_common_date})')
plt.xlabel('Monthly Rent')
plt.ylabel('House Price')
plt.grid(True)
plt.show()

# %% # Calculate Regression Residuals and Identify Outliers
print("# %% # Calculate Regression Residuals and Identify Outliers")
# Calculate the predicted prices based on the linear regression
# We need to fit a model to get the parameters first (slope and intercept)
# Using numpy.polyfit for simplicity
rent_values = merged_df_cleaned['Rent'].values
price_values = merged_df_cleaned['Price'].values

slope, intercept = np.polyfit(rent_values, price_values, 1) # 1 for linear fit

merged_df_cleaned['Predicted_Price'] = intercept + slope * merged_df_cleaned['Rent']
merged_df_cleaned['Residual'] = merged_df_cleaned['Price'] - merged_df_cleaned['Predicted_Price']

# Ensure RegionName_str exists for merging with Census data later
if 'RegionName_str' not in merged_df_cleaned.columns and 'RegionName' in merged_df_cleaned.columns:
    merged_df_cleaned['RegionName_str'] = merged_df_cleaned['RegionName'].astype(str).str.zfill(5)
elif 'RegionName' not in merged_df_cleaned.columns:
    print("Warning: 'RegionName' column not found in merged_df_cleaned. Cannot create 'RegionName_str'.")
    # Handle error or add placeholder if necessary, for now, it might affect the merge below.

# Sort by residual to find outliers
residuals_sorted_df = merged_df_cleaned.sort_values(by='Residual', ascending=False)

# Merge rent_share information into the residuals_sorted_df
# Ensure df_screened has 'zcta' and 'rent_share' for the merge
if 'zcta' in df_screened.columns and 'rent_share' in df_screened.columns:
    residuals_sorted_df = pd.merge(
        residuals_sorted_df,
        df_screened[['zcta', 'rent_share']],
        left_on='RegionName_str',  # This was created in the merge cell for Zillow+Census
        right_on='zcta',
        how='left'
    )
    # We can drop the redundant 'zcta' column if 'RegionName_str' or 'RegionName' is preferred for display
    # For now, let's keep it and decide on display columns carefully
else:
    print("Warning: 'zcta' or 'rent_share' not found in df_screened. Cannot add rent_share to residuals table.")
    residuals_sorted_df['rent_share'] = np.nan # Add a placeholder column if merge fails


print("Top 5 Zip Codes with Prices Unusually High for Rents (Highest Positive Residuals):")
print(residuals_sorted_df[['RegionName', 'City', 'State', 'Metro', 'CountyName', 'Price', 'Rent', 'Predicted_Price', 'Residual', 'rent_share']].head(5))

print("\nTop 5 Zip Codes with Prices Unusually Low for Rents (Highest Negative Residuals):")
print(residuals_sorted_df[['RegionName', 'City', 'State', 'Metro', 'CountyName', 'Price', 'Rent', 'Predicted_Price', 'Residual', 'rent_share']].tail(5))

# %% Census
# ---------------------------------------------------------------------------
# Pull ACS-5-year 2022 table B25003 for every ZCTA and compute renter share
# ---------------------------------------------------------------------------
import requests, pandas as pd

CENSUS_API_KEY = "06048dc3bd32068702b5ef9b49875ec0c5ca56ce"   # your key
YEAR           = "2022"
BASE_URL       = f"https://api.census.gov/data/{YEAR}/acs/acs5"

# variables: owner-occupied (002E) & renter-occupied (003E) units
VAR_OWNER  = "B25003_002E"
VAR_RENTER = "B25003_003E"

params = {
    "get":  f"{VAR_OWNER},{VAR_RENTER}",          # you don't need NAME here
    "for":  "zip code tabulation area:*",
    "key":  CENSUS_API_KEY
}

# --- API CALL ---------------------------------------------------------------
data = requests.get(BASE_URL, params=params).json()   # returns list-of-lists
df   = pd.DataFrame(data[1:], columns=data[0])        # first sub-list is header

# --- CLEAN & CAST -----------------------------------------------------------
df = df.rename(columns={
    VAR_OWNER:  "owner_occ",
    VAR_RENTER: "renter_occ",
    "zip code tabulation area": "zcta"
})

df[["owner_occ", "renter_occ"]] = df[["owner_occ", "renter_occ"]].apply(pd.to_numeric)
df["zcta"] = df["zcta"].str.zfill(5)

# --- DERIVED METRICS & OPTIONAL SCREEN --------------------------------------
df["total_occ"]  = df["owner_occ"] + df["renter_occ"]
df["rent_share"] = df["renter_occ"] / df["total_occ"]

min_rent_share = 0.10   # 10 % rentals
min_total      = 200    # at least 200 occupied units

df_screened = df.loc[(df["rent_share"] >= min_rent_share) &
                     (df["total_occ"]  >= min_total)]

# ready to merge on ZIP / RegionName
df_screened.head()

# %% # Merge Zillow Data with Census Renter Share Data
print("# %% # Merge Zillow Data with Census Renter Share Data")

# Ensure RegionName in merged_df_cleaned is a string and padded to 5 digits for merging with zcta
# Assuming RegionName is originally numeric (e.g., an integer zip code)
# Create a new column for the string version to ensure clean merge key, if it doesn't exist
if 'RegionName_str' not in merged_df_cleaned.columns:
    merged_df_cleaned['RegionName_str'] = merged_df_cleaned['RegionName'].astype(str).str.zfill(5)

# df_screened has 'zcta' as the 5-digit string zip code from your census script
merged_with_census_df = pd.merge(
    merged_df_cleaned,
    df_screened,
    left_on='RegionName_str', # Use the formatted string zip from Zillow data
    right_on='zcta',          # Use the zcta from Census data
    how='inner'               # Keep only zip codes present in both
)

print(f"Shape of merged_df_cleaned: {merged_df_cleaned.shape}")
print(f"Shape of df_screened: {df_screened.shape}")
print(f"Shape of merged_with_census_df: {merged_with_census_df.shape}")
print("Columns in merged DataFrame:", merged_with_census_df.columns)
print("Head of merged DataFrame with key columns:")
print(merged_with_census_df[['RegionName', 'RegionName_str', 'zcta', 'City', 'State', 'rent_share', 'Price', 'Rent', 'Price_to_Rent_Ratio']].head())

# %% # Display Top and Bottom 10 Zip Codes by Renter Share
print("# %% # Display Top and Bottom 10 Zip Codes by Renter Share")

# Sort the DataFrame by rent_share
# Make sure 'rent_share' column is numeric if it's not already (it should be from your census script)
merged_with_census_df['rent_share'] = pd.to_numeric(merged_with_census_df['rent_share'], errors='coerce')
merged_with_census_df_sorted_by_rent_share = merged_with_census_df.sort_values(by='rent_share', ascending=False).dropna(subset=['rent_share'])

# Define columns to display - using 'zcta' as the standardized zip code string
cols_to_display = [
    'zcta', 'City', 'State', 'Metro', 'CountyName',
    'Price', 'Rent', 'Price_to_Rent_Ratio', 'rent_share'
]

print("\nTop 10 Zip Codes by Renter Share (Highest % Renters):")
print(merged_with_census_df_sorted_by_rent_share[cols_to_display].head(10))

print("\nBottom 10 Zip Codes by Renter Share (Lowest % Renters):")
print(merged_with_census_df_sorted_by_rent_share[cols_to_display].tail(10))

# %% # Visualize Price-to-Rent Ratio vs. Renter Share
print("# %% # Visualize Price-to-Rent Ratio vs. Renter Share")

# Ensure Price_to_Rent_Ratio is numeric and finite for plotting
plot_df = merged_with_census_df.copy()
plot_df['Price_to_Rent_Ratio'] = pd.to_numeric(plot_df['Price_to_Rent_Ratio'], errors='coerce')
plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['rent_share', 'Price_to_Rent_Ratio'])

plt.figure(figsize=(12, 8))
sns.scatterplot(data=plot_df, x='rent_share', y='Price_to_Rent_Ratio', alpha=0.5)

# Optional: Add a vertical line at a specific rent_share threshold, e.g., 15%
plt.axvline(x=0.15, color='red', linestyle='--', label='15% Renter Share')

plt.title('Price-to-Rent Ratio vs. Renter Share')
plt.xlabel('Renter Share (from Census)')
plt.ylabel('Price-to-Rent Ratio')
plt.legend()
plt.grid(True)
plt.show()

# %% # Scatter Plot: Price vs. Rent (All data, Rent <= $10k)
print("# %% # Scatter Plot: Price vs. Rent (All data, Rent <= $10k)")

# Ensure merged_df_cleaned is available and columns are numeric/finite
# This df is already filtered for Rent <= 10000 and cleaned from previous cells
if 'merged_df_cleaned' in locals() and isinstance(merged_df_cleaned, pd.DataFrame):
    plot_data_all = merged_df_cleaned.copy()
    plot_data_all['Price'] = pd.to_numeric(plot_data_all['Price'], errors='coerce')
    plot_data_all['Rent'] = pd.to_numeric(plot_data_all['Rent'], errors='coerce')
    plot_data_all = plot_data_all.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_data_all, x='Rent', y='Price', alpha=0.5)
    sns.regplot(data=plot_data_all, x='Rent', y='Price', scatter=False, color='red')
    plt.title(f'Price vs. Rent (All Zip Codes with Rent <= $10,000, Latest Month: {latest_common_date})')
    plt.xlabel('Monthly Rent')
    plt.ylabel('House Price')
    plt.grid(True)
    plt.show()
else:
    print("merged_df_cleaned not found or not a DataFrame. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent (Price <= $5,000,000)
print("# %% # Scatter Plot: Price vs. Rent (Price <= $5,000,000)")
price_cutoff_5M = 5_000_000
if 'merged_df_cleaned' in locals() and isinstance(merged_df_cleaned, pd.DataFrame):
    plot_data_5M = merged_df_cleaned[merged_df_cleaned['Price'] <= price_cutoff_5M].copy()
    plot_data_5M['Price'] = pd.to_numeric(plot_data_5M['Price'], errors='coerce')
    plot_data_5M['Rent'] = pd.to_numeric(plot_data_5M['Rent'], errors='coerce')
    plot_data_5M = plot_data_5M.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not plot_data_5M.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_data_5M, x='Rent', y='Price', alpha=0.5)
        sns.regplot(data=plot_data_5M, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent (Price <= ${price_cutoff_5M:,}, Rent <= $10,000, Latest Month: {latest_common_date})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for Price <= ${price_cutoff_5M:,} (and Rent <= $10,000).")
else:
    print("merged_df_cleaned not found. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent (Price <= $4,000,000)
print("# %% # Scatter Plot: Price vs. Rent (Price <= $4,000,000)")
price_cutoff_4M = 4_000_000
if 'merged_df_cleaned' in locals() and isinstance(merged_df_cleaned, pd.DataFrame):
    plot_data_4M = merged_df_cleaned[merged_df_cleaned['Price'] <= price_cutoff_4M].copy()
    plot_data_4M['Price'] = pd.to_numeric(plot_data_4M['Price'], errors='coerce')
    plot_data_4M['Rent'] = pd.to_numeric(plot_data_4M['Rent'], errors='coerce')
    plot_data_4M = plot_data_4M.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not plot_data_4M.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_data_4M, x='Rent', y='Price', alpha=0.5)
        sns.regplot(data=plot_data_4M, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent (Price <= ${price_cutoff_4M:,}, Rent <= $10,000, Latest Month: {latest_common_date})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for Price <= ${price_cutoff_4M:,} (and Rent <= $10,000).")
else:
    print("merged_df_cleaned not found. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent (Price <= $3,000,000)
print("# %% # Scatter Plot: Price vs. Rent (Price <= $3,000,000)")
price_cutoff_3M = 3_000_000
if 'merged_df_cleaned' in locals() and isinstance(merged_df_cleaned, pd.DataFrame):
    plot_data_3M = merged_df_cleaned[merged_df_cleaned['Price'] <= price_cutoff_3M].copy()
    plot_data_3M['Price'] = pd.to_numeric(plot_data_3M['Price'], errors='coerce')
    plot_data_3M['Rent'] = pd.to_numeric(plot_data_3M['Rent'], errors='coerce')
    plot_data_3M = plot_data_3M.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not plot_data_3M.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_data_3M, x='Rent', y='Price', alpha=0.5)
        sns.regplot(data=plot_data_3M, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent (Price <= ${price_cutoff_3M:,}, Rent <= $10,000, Latest Month: {latest_common_date})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for Price <= ${price_cutoff_3M:,} (and Rent <= $10,000).")
else:
    print("merged_df_cleaned not found. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent (Price <= $2,000,000)
print("# %% # Scatter Plot: Price vs. Rent (Price <= $2,000,000)")
price_cutoff_2M = 2_000_000
if 'merged_df_cleaned' in locals() and isinstance(merged_df_cleaned, pd.DataFrame):
    plot_data_2M = merged_df_cleaned[merged_df_cleaned['Price'] <= price_cutoff_2M].copy()
    plot_data_2M['Price'] = pd.to_numeric(plot_data_2M['Price'], errors='coerce')
    plot_data_2M['Rent'] = pd.to_numeric(plot_data_2M['Rent'], errors='coerce')
    plot_data_2M = plot_data_2M.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not plot_data_2M.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_data_2M, x='Rent', y='Price', alpha=0.5)
        sns.regplot(data=plot_data_2M, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent (Price <= ${price_cutoff_2M:,}, Rent <= $10,000, Latest Month: {latest_common_date})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for Price <= ${price_cutoff_2M:,} (and Rent <= $10,000).")
else:
    print("merged_df_cleaned not found. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent (Price <= $1,000,000)
print("# %% # Scatter Plot: Price vs. Rent (Price <= $1,000,000)")
price_cutoff_1M = 1_000_000
if 'merged_df_cleaned' in locals() and isinstance(merged_df_cleaned, pd.DataFrame):
    plot_data_1M = merged_df_cleaned[merged_df_cleaned['Price'] <= price_cutoff_1M].copy()
    plot_data_1M['Price'] = pd.to_numeric(plot_data_1M['Price'], errors='coerce')
    plot_data_1M['Rent'] = pd.to_numeric(plot_data_1M['Rent'], errors='coerce')
    plot_data_1M = plot_data_1M.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not plot_data_1M.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_data_1M, x='Rent', y='Price', alpha=0.5)
        sns.regplot(data=plot_data_1M, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent (Price <= ${price_cutoff_1M:,}, Rent <= $10,000, Latest Month: {latest_common_date})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for Price <= ${price_cutoff_1M:,} (and Rent <= $10,000).")
else:
    print("merged_df_cleaned not found. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price-to-Rent Ratio vs. Total Occupancy
print("# %% # Scatter Plot: Price-to-Rent Ratio vs. Total Occupancy")

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    plot_occupancy_df = merged_with_census_df.copy()
    
    # Ensure necessary columns are numeric and finite
    plot_occupancy_df['total_occ'] = pd.to_numeric(plot_occupancy_df['total_occ'], errors='coerce')
    plot_occupancy_df['Price_to_Rent_Ratio'] = pd.to_numeric(plot_occupancy_df['Price_to_Rent_Ratio'], errors='coerce')
    plot_occupancy_df = plot_occupancy_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['total_occ', 'Price_to_Rent_Ratio'])

    if not plot_occupancy_df.empty:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=plot_occupancy_df, x='total_occ', y='Price_to_Rent_Ratio', alpha=0.5)
        
        plt.title('Price-to-Rent Ratio vs. Total Occupied Units')
        plt.xlabel('Total Occupied Units (Owner + Renter Households)')
        plt.ylabel('Price-to-Rent Ratio')
        plt.grid(True)
        plt.show()
    else:
        print("No data available after cleaning for Price-to-Rent Ratio vs. Total Occupancy plot.")
else:
    print("merged_with_census_df not found or not a DataFrame. Please ensure previous cells have run correctly.")

# %% # Histogram of Total Occupied Units
print("# %% # Histogram of Total Occupied Units")

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    hist_df = merged_with_census_df.copy()
    
    # Ensure total_occ is numeric and finite
    hist_df['total_occ'] = pd.to_numeric(hist_df['total_occ'], errors='coerce')
    hist_df = hist_df.dropna(subset=['total_occ'])

    if not hist_df.empty:
        plt.figure(figsize=(12, 7))
        sns.histplot(data=hist_df, x='total_occ', bins=50, kde=True) # Using 50 bins, can be adjusted
        
        plt.title('Distribution of Total Occupied Units per Zip Code')
        plt.xlabel('Total Occupied Units (Owner + Renter Households)')
        plt.ylabel('Frequency (Number of Zip Codes)')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        
        # Print descriptive statistics for total_occ
        print("\nDescriptive statistics for Total Occupied Units:")
        print(hist_df['total_occ'].describe())
    else:
        print("No data available for Total Occupied Units histogram after cleaning.")
else:
    print("merged_with_census_df not found or not a DataFrame. Please ensure previous cells have run correctly.")

# %% # Histogram of Renter Share
print("# %% # Histogram of Renter Share")

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    hist_rent_share_df = merged_with_census_df.copy()
    
    # Ensure rent_share is numeric and finite
    hist_rent_share_df['rent_share'] = pd.to_numeric(hist_rent_share_df['rent_share'], errors='coerce')
    hist_rent_share_df = hist_rent_share_df.dropna(subset=['rent_share'])

    if not hist_rent_share_df.empty:
        plt.figure(figsize=(12, 7))
        # Renter share is a proportion, so bins might be better defined or more numerous for detail
        sns.histplot(data=hist_rent_share_df, x='rent_share', bins=30, kde=True) 
        
        plt.title('Distribution of Renter Share per Zip Code')
        plt.xlabel('Renter Share (Proportion of Renter-Occupied Households)')
        plt.ylabel('Frequency (Number of Zip Codes)')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        
        # Print descriptive statistics for rent_share
        print("\nDescriptive statistics for Renter Share:")
        print(hist_rent_share_df['rent_share'].describe())
    else:
        print("No data available for Renter Share histogram after cleaning.")
else:
    print("merged_with_census_df not found or not a DataFrame. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent for California (Filtered)
print("# %% # Scatter Plot: Price vs. Rent for California (Filtered)")

state_code = 'CA'
price_cap = 2_000_000
min_rent_share = 0.20
min_total_occ = 5_000

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    df_filtered_state = merged_with_census_df[
        (merged_with_census_df['Price'] <= price_cap) &
        (merged_with_census_df['rent_share'] > min_rent_share) &
        (merged_with_census_df['total_occ'] > min_total_occ) &
        (merged_with_census_df['State'] == state_code)
    ].copy()
    
    # Ensure Price and Rent are numeric and finite for plotting
    df_filtered_state['Price'] = pd.to_numeric(df_filtered_state['Price'], errors='coerce')
    df_filtered_state['Rent'] = pd.to_numeric(df_filtered_state['Rent'], errors='coerce')
    df_filtered_state = df_filtered_state.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not df_filtered_state.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_filtered_state, x='Rent', y='Price', alpha=0.6)
        sns.regplot(data=df_filtered_state, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent in {state_code}\n(Price <= ${price_cap:,}, Rent Share > {min_rent_share:.0%}, Total Occupied > {min_total_occ:,})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for {state_code} with the specified filters.")
else:
    print("merged_with_census_df not found. Please ensure previous cells have run correctly.")

# %% # Scatter Plot: Price vs. Rent for New York (Filtered)
print("# %% # Scatter Plot: Price vs. Rent for New York (Filtered)")

state_code = 'NY'
price_cap = 2_000_000
min_rent_share = 0.20
min_total_occ = 5_000

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    df_filtered_state = merged_with_census_df[
        (merged_with_census_df['Price'] <= price_cap) &
        (merged_with_census_df['rent_share'] > min_rent_share) &
        (merged_with_census_df['total_occ'] > min_total_occ) &
        (merged_with_census_df['State'] == state_code)
    ].copy()
    
    df_filtered_state['Price'] = pd.to_numeric(df_filtered_state['Price'], errors='coerce')
    df_filtered_state['Rent'] = pd.to_numeric(df_filtered_state['Rent'], errors='coerce')
    df_filtered_state = df_filtered_state.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not df_filtered_state.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_filtered_state, x='Rent', y='Price', alpha=0.6)
        sns.regplot(data=df_filtered_state, x='Rent', y='Price', scatter=False, color='red')
        plt.title(f'Price vs. Rent in {state_code}\n(Price <= ${price_cap:,}, Rent Share > {min_rent_share:.0%}, Total Occupied > {min_total_occ:,})')
        plt.xlabel('Monthly Rent')
        plt.ylabel('House Price')
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for {state_code} with the specified filters.")
else:
    print("merged_with_census_df not found. Please ensure previous cells have run correctly.")

# %% # INTERACTIVE Scatter Plot: Price vs. Rent for New York (Filtered)
print("# %% # INTERACTIVE Scatter Plot: Price vs. Rent for New York (Filtered)")

interactive_state_code_ny = 'NY'
interactive_price_cap_ny = 2_000_000
interactive_min_rent_share_ny = 0.20
interactive_min_total_occ_ny = 5_000

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    df_interactive_ny = merged_with_census_df[
        (merged_with_census_df['Price'] <= interactive_price_cap_ny) &
        (merged_with_census_df['rent_share'] > interactive_min_rent_share_ny) &
        (merged_with_census_df['total_occ'] > interactive_min_total_occ_ny) &
        (merged_with_census_df['State'] == interactive_state_code_ny)
    ].copy()
    
    # Ensure Price and Rent are numeric and finite for plotting
    df_interactive_ny['Price'] = pd.to_numeric(df_interactive_ny['Price'], errors='coerce')
    df_interactive_ny['Rent'] = pd.to_numeric(df_interactive_ny['Rent'], errors='coerce')
    df_interactive_ny = df_interactive_ny.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent'])

    if not df_interactive_ny.empty:
        # Define columns for hover information
        # Assuming RegionName is the zip, and other geo columns are present
        hover_columns_ny = ['RegionName', 'City', 'CountyName', 'State', 'Price', 'Rent', 'Price_to_Rent_Ratio', 'rent_share', 'total_occ']
        
        # Ensure hover columns exist in the dataframe, if not, they won't show in hover (or handle missing ones)
        actual_hover_cols_ny = [col for col in hover_columns_ny if col in df_interactive_ny.columns]

        fig_interactive_ny = px.scatter(
            df_interactive_ny,
            x='Rent',
            y='Price',
            trendline="ols",  # Ordinary Least Squares regression line
            opacity=0.7,
            hover_data=actual_hover_cols_ny, # Use only existing columns
            title=f'<b>Interactive:</b> Price vs. Rent in {interactive_state_code_ny}<br>'
                  f'(Price <= ${interactive_price_cap_ny:,}, Rent Share > {interactive_min_rent_share_ny:.0%}, Total Occupied > {interactive_min_total_occ_ny:,})'
        )
        fig_interactive_ny.update_layout(xaxis_title='Monthly Rent', yaxis_title='House Price')
        fig_interactive_ny.show()
    else:
        print(f"No data available for {interactive_state_code_ny} with the specified filters for the interactive plot.")
else:
    print("merged_with_census_df not found. Please ensure previous cells have run correctly for the interactive plot.")

# %% # INTERACTIVE Scatter Plot: Price vs. Rent for New York (Filtered) - New Cell
print("# %% # INTERACTIVE Scatter Plot: Price vs. Rent for New York (Filtered) - New Cell")

interactive_state_code_ny_new = 'NY'
interactive_price_cap_ny_new = 2_000_000
interactive_min_rent_share_ny_new = 0.20
interactive_min_total_occ_ny_new = 5_000

if 'merged_with_census_df' in locals() and isinstance(merged_with_census_df, pd.DataFrame):
    df_interactive_ny_new = merged_with_census_df[
        (merged_with_census_df['Price'] <= interactive_price_cap_ny_new) &
        (merged_with_census_df['rent_share'] > interactive_min_rent_share_ny_new) &
        (merged_with_census_df['total_occ'] > interactive_min_total_occ_ny_new) &
        (merged_with_census_df['State'] == interactive_state_code_ny_new)
    ].copy()
    
    df_interactive_ny_new['Price'] = pd.to_numeric(df_interactive_ny_new['Price'], errors='coerce')
    df_interactive_ny_new['Rent'] = pd.to_numeric(df_interactive_ny_new['Rent'], errors='coerce')
    df_interactive_ny_new['rent_share'] = pd.to_numeric(df_interactive_ny_new['rent_share'], errors='coerce')
    df_interactive_ny_new['total_occ'] = pd.to_numeric(df_interactive_ny_new['total_occ'], errors='coerce')
    
    df_interactive_ny_new = df_interactive_ny_new.replace([np.inf, -np.inf], np.nan).dropna(subset=['Price', 'Rent', 'rent_share', 'total_occ'])

    if not df_interactive_ny_new.empty and df_interactive_ny_new['rent_share'].nunique() >= 5:
        # Create quintiles for rent_share
        # Ensure there are enough unique values for 5 quintiles, otherwise pd.qcut might fail or produce fewer bins
        try:
            df_interactive_ny_new['Rent Share Quintile'] = pd.qcut(df_interactive_ny_new['rent_share'], 5, labels=[f"Quintile {i+1}" for i in range(5)])
        except ValueError:
             # If not enough unique values for 5 quintiles, try fewer or handle as an error/warning
            print("Warning: Could not create 5 distinct quintiles for rent_share. Plotting without quintile coloring.")
            df_interactive_ny_new['Rent Share Quintile'] = "N/A"
            # Fallback: or remove quintile coloring from the plot call below

        hover_columns_ny_new = ['RegionName', 'City', 'CountyName', 'State', 'Price', 'Rent', 'Price_to_Rent_Ratio', 'rent_share', 'total_occ', 'Rent Share Quintile']
        actual_hover_cols_ny_new = [col for col in hover_columns_ny_new if col in df_interactive_ny_new.columns]

        fig_interactive_ny_new = px.scatter(
            df_interactive_ny_new,
            x='Rent',
            y='Price',
            color='Rent Share Quintile', # Color by renter share quintile
            size='total_occ',
            trendline="ols",
            opacity=0.7,
            hover_data=actual_hover_cols_ny_new,
            category_orders={"Rent Share Quintile": [f"Quintile {i+1}" for i in range(5)]}, # Ensures legend order
            title=f'<b>Interactive:</b> Price vs. Rent in {interactive_state_code_ny_new} (Colored by Rent Share Quintile)<br>'
                  f'(Price <= ${interactive_price_cap_ny_new:,}, Rent Share > {interactive_min_rent_share_ny_new:.0%}, Total Occupied > {interactive_min_total_occ_ny_new:,})'
        )
        fig_interactive_ny_new.update_layout(xaxis_title='Monthly Rent', yaxis_title='House Price')
        fig_interactive_ny_new.show()
    elif df_interactive_ny_new.empty:
        print(f"No data available for {interactive_state_code_ny_new} with the specified filters for the new interactive plot.")
    else: # Not empty, but not enough unique values for quintiles
        print("Not enough unique rent_share values to create 5 quintiles. Plotting without quintile-based coloring.")
        # Optionally, you could plot here without the quintile coloring as a fallback
        # For now, it will just print this message and not plot if quintiles can't be made.

else:
    print("merged_with_census_df not found. Please ensure previous cells have run correctly for the new interactive plot.")

# %% # Load Apartment Price Data (ZHVI Condo Tier)
print("# %% # Load Apartment Price Data (ZHVI Condo Tier)")

try:
    Price_apartments = pd.read_csv('../ZillowData/Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv', encoding='latin-1')
    print("Price_apartments DataFrame loaded successfully.")
    print(Price_apartments.head())
except FileNotFoundError:
    print("Error: '../ZillowData/Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv' not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred while loading Price_apartments: {e}")

# %% # Calculate Apartment Price-to-Rent Ratio
print("# %% # Calculate Apartment Price-to-Rent Ratio")

if 'Price_apartments' in locals() and 'rent' in locals():
    # Identify date columns in Price_apartments
    # Assuming the same date format as the original price DataFrame (YYYY-MM-DD)
    apartment_price_date_cols = [col for col in Price_apartments.columns if isinstance(col, str) and col.count('-') == 2 and len(col) == 10]
    rent_date_cols_ap = [col for col in rent.columns if isinstance(col, str) and col.count('-') == 2 and len(col) == 10]

    if not apartment_price_date_cols:
        print("Error: No date columns found in Price_apartments. Please check column names/format.")
    elif not rent_date_cols_ap: # Should be same as before, but good to check in context
        print("Error: No date columns found in rent DataFrame. Please check column names/format.")
    else:
        latest_apartment_price_date = max(apartment_price_date_cols)
        latest_rent_date_for_ap = max(rent_date_cols_ap) # This will be the same as latest_rent_date found earlier
        
        latest_common_date_ap = min(latest_apartment_price_date, latest_rent_date_for_ap)
        print(f"Latest common date for Apartment Price and Rent data: {latest_common_date_ap}")

        # Select relevant columns from Price_apartments
        # Assuming identifier columns are the same as in the original 'price' DataFrame
        identifier_cols_ap = ['RegionName', 'City', 'State', 'Metro', 'CountyName']
        # Check if all identifier_cols_ap are in Price_apartments.columns
        missing_cols = [col for col in identifier_cols_ap if col not in Price_apartments.columns]
        if missing_cols:
            print(f"Warning: The following identifier columns are missing in Price_apartments and will be excluded: {missing_cols}")
            identifier_cols_ap = [col for col in identifier_cols_ap if col in Price_apartments.columns]
            if 'RegionName' not in identifier_cols_ap:
                 print("Error: 'RegionName' is crucial and missing from Price_apartments. Cannot proceed with merge.")
                 # Set a flag or exit if RegionName is critical and missing
                 can_proceed_ap = False
            else:
                can_proceed_ap = True
        else:
            can_proceed_ap = True
        
        if can_proceed_ap:
            price_apartments_info_latest = Price_apartments[identifier_cols_ap + [latest_common_date_ap]].rename(columns={latest_common_date_ap: 'Apartment_Price'})
            rent_values_ap_latest = rent[['RegionName'] + [latest_common_date_ap]].rename(columns={latest_common_date_ap: 'Rent'})

            # Merge the two dataframes on RegionName
            merged_apartments_df = pd.merge(price_apartments_info_latest, rent_values_ap_latest, on='RegionName', how='inner')

            # Calculate the Apartment Price-to-Rent Ratio
            merged_apartments_df['Apartment_Price_to_Rent_Ratio'] = merged_apartments_df['Apartment_Price'] / merged_apartments_df['Rent']

            print("\nHead of merged_apartments_df:")
            print(merged_apartments_df.head())
            print("\nSummary statistics for Apartment_Price_to_Rent_Ratio:")
            print(merged_apartments_df['Apartment_Price_to_Rent_Ratio'].describe())
        else:
            print("Could not proceed with creating merged_apartments_df due to missing crucial columns.")
else:
    print("Error: Price_apartments or rent DataFrame not found. Please ensure they are loaded correctly.")

# %% # Display Top and Bottom 10 Zip Codes by Apartment Price-to-Rent Ratio
print("# %% # Display Top and Bottom 10 Zip Codes by Apartment Price-to-Rent Ratio")

if 'merged_apartments_df' in locals() and isinstance(merged_apartments_df, pd.DataFrame):
    # Ensure the ratio column is numeric and clean for sorting
    merged_apartments_df['Apartment_Price_to_Rent_Ratio'] = pd.to_numeric(merged_apartments_df['Apartment_Price_to_Rent_Ratio'], errors='coerce')
    merged_apartments_df_cleaned_for_ratio_display = merged_apartments_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Apartment_Price_to_Rent_Ratio'])

    if not merged_apartments_df_cleaned_for_ratio_display.empty:
        merged_apartments_sorted = merged_apartments_df_cleaned_for_ratio_display.sort_values(by='Apartment_Price_to_Rent_Ratio', ascending=False)

        # Define columns to display, checking if they exist in the dataframe
        # Start with a base set and add geo identifiers if they are present
        display_cols_ap_ratio = ['RegionName', 'Apartment_Price', 'Rent', 'Apartment_Price_to_Rent_Ratio']
        possible_geo_cols = ['City', 'State', 'Metro', 'CountyName']
        for col in possible_geo_cols:
            if col in merged_apartments_sorted.columns:
                # Insert geo columns after RegionName for better readability
                display_cols_ap_ratio.insert(1, col)
        # Remove duplicates if any column was added more than once (though list.insert doesn't cause this)
        display_cols_ap_ratio = sorted(set(display_cols_ap_ratio), key=display_cols_ap_ratio.index)
        
        # Ensure all selected display_cols_ap_ratio actually exist to prevent KeyErrors
        final_display_cols = [col for col in display_cols_ap_ratio if col in merged_apartments_sorted.columns]

        print("\nTop 10 Zip Codes by Apartment Price-to-Rent Ratio:")
        print(merged_apartments_sorted[final_display_cols].head(10))

        print("\nBottom 10 Zip Codes by Apartment Price-to-Rent Ratio:")
        print(merged_apartments_sorted[final_display_cols].tail(10))
    else:
        print("No valid data to display for Apartment Price-to-Rent Ratio (after cleaning NaNs/Infs).")
else:
    print("merged_apartments_df not found. Please run the previous cell to generate it.")

# %% # Scatter Plot: Apartment Price vs. Rent (Rent <= $10k)
# Becomes: Log-Log Apartment Price vs. Rent (Rent <= $4k, Color by Rent Share, Size by Occupancy)
print("# %% # Log-Log Scatter Plot: Apartment Price vs. Rent (Rent <= $4k, Color by Rent Share, Size by Occupancy)")

if 'merged_apartments_df' in locals() and isinstance(merged_apartments_df, pd.DataFrame):
    merged_apartments_df_cleaned = merged_apartments_df.copy()
    merged_apartments_df_cleaned['Apartment_Price'] = pd.to_numeric(merged_apartments_df_cleaned['Apartment_Price'], errors='coerce')
    merged_apartments_df_cleaned['Rent'] = pd.to_numeric(merged_apartments_df_cleaned['Rent'], errors='coerce')
    
    rent_cap_ap = 4000 # Updated rent cap
    merged_apartments_df_cleaned = merged_apartments_df_cleaned[merged_apartments_df_cleaned['Rent'] <= rent_cap_ap]
    merged_apartments_df_cleaned = merged_apartments_df_cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=['Apartment_Price', 'Rent'])

    # Merge with Census data (df_screened) to get rent_share and total_occ
    merged_apartments_plot_df = pd.DataFrame() # Initialize
    if 'df_screened' in locals() and isinstance(df_screened, pd.DataFrame):
        # Ensure RegionName_str for merging in merged_apartments_df_cleaned
        if 'RegionName_str' not in merged_apartments_df_cleaned.columns and 'RegionName' in merged_apartments_df_cleaned.columns:
            merged_apartments_df_cleaned['RegionName_str'] = merged_apartments_df_cleaned['RegionName'].astype(str).str.zfill(5)
        elif 'RegionName' not in merged_apartments_df_cleaned.columns:
            print("Warning: 'RegionName' missing from apartment data, cannot create 'RegionName_str' for merge.")
        
        if 'RegionName_str' in merged_apartments_df_cleaned.columns and 'zcta' in df_screened.columns:
            merged_apartments_plot_df = pd.merge(
                merged_apartments_df_cleaned,
                df_screened[['zcta', 'rent_share', 'total_occ']],
                left_on='RegionName_str',
                right_on='zcta',
                how='left'  # Keep all apartment data points
            )
            merged_apartments_plot_df['rent_share'] = pd.to_numeric(merged_apartments_plot_df['rent_share'], errors='coerce')
            merged_apartments_plot_df['total_occ'] = pd.to_numeric(merged_apartments_plot_df['total_occ'], errors='coerce')
            # Drop rows if rent_share or total_occ are NaN, as they are used for hue/size
            merged_apartments_plot_df.dropna(subset=['rent_share', 'total_occ'], inplace=True)
        else:
            print("Warning: Merge keys ('RegionName_str' or 'zcta') missing. Proceeding without census data for aesthetics.")
            merged_apartments_plot_df = merged_apartments_df_cleaned.copy()
    else:
        print("Warning: df_screened (Census data) not found. Cannot color by rent_share or size by total_occ.")
        merged_apartments_plot_df = merged_apartments_df_cleaned.copy()

    if not merged_apartments_plot_df.empty:
        plt.figure(figsize=(14, 10))
        
        hue_col = 'rent_share' if 'rent_share' in merged_apartments_plot_df.columns else None
        size_col = 'total_occ' if 'total_occ' in merged_apartments_plot_df.columns else None

        sns.scatterplot(
            data=merged_apartments_plot_df, 
            x='Rent', 
            y='Apartment_Price', 
            alpha=0.6,
            hue=hue_col,
            size=size_col,
            sizes=(30, 400) # Adjust size range as needed
        )
        
        # Set axes to log scale BEFORE calling regplot if we want it to draw on log axes
        plt.xscale('log')
        plt.yscale('log')

        # Add the regression line. It will fit on original data but display on log axes.
        sns.regplot(
            data=merged_apartments_plot_df, 
            x='Rent', 
            y='Apartment_Price', 
            scatter=False, 
            color='darkred',
            line_kws={'linestyle':'--'}
        )
        
        title_date_ap = latest_common_date_ap if 'latest_common_date_ap' in locals() else "Latest Month"
        plt.title(f'Log-Log Apartment Price vs. Rent (Rent <= ${rent_cap_ap:,}, {title_date_ap})\nColored by Rent Share, Sized by Total Occupancy')
        plt.xlabel('Monthly Rent (Log Scale)')
        plt.ylabel('Apartment Price (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        
        if hue_col:
            plt.legend(title=hue_col.replace('_',' ').title(), loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend if it's outside
        plt.show()
    else:
        print("No data available for Apartment Price vs. Rent scatter plot after cleaning, filtering, and merging.")
else:
    print("merged_apartments_df not found. Please run the cell where it is generated.")

# %% # Calculate Regression Residuals for Apartment Price vs. Rent and Identify Outliers
print("# %% # Calculate Regression Residuals for Apartment Price vs. Rent and Identify Outliers")

if 'merged_apartments_df_cleaned' in locals() and isinstance(merged_apartments_df_cleaned, pd.DataFrame) and not merged_apartments_df_cleaned.empty:
    # Calculate the predicted prices based on the linear regression
    ap_rent_values = merged_apartments_df_cleaned['Rent'].values
    ap_price_values = merged_apartments_df_cleaned['Apartment_Price'].values

    # Ensure there's enough data to fit a line
    if len(ap_rent_values) > 1 and len(ap_price_values) > 1:
        ap_slope, ap_intercept = np.polyfit(ap_rent_values, ap_price_values, 1) # Linear fit

        merged_apartments_df_cleaned['Predicted_Apartment_Price'] = ap_intercept + ap_slope * merged_apartments_df_cleaned['Rent']
        merged_apartments_df_cleaned['Apartment_Price_Residual'] = merged_apartments_df_cleaned['Apartment_Price'] - merged_apartments_df_cleaned['Predicted_Apartment_Price']

        # Sort by residual to find outliers
        ap_residuals_sorted_df = merged_apartments_df_cleaned.sort_values(by='Apartment_Price_Residual', ascending=False)
        
        # Define columns for display, checking for existence
        base_ap_resid_cols = ['RegionName', 'Apartment_Price', 'Rent', 'Predicted_Apartment_Price', 'Apartment_Price_Residual', 'Apartment_Price_to_Rent_Ratio']
        geo_cols_ap_resid = ['City', 'State', 'Metro', 'CountyName']
        
        display_cols_ap_resid = []
        if 'RegionName' in ap_residuals_sorted_df.columns: display_cols_ap_resid.append('RegionName')
        for col in geo_cols_ap_resid:
            if col in ap_residuals_sorted_df.columns: display_cols_ap_resid.append(col)
        for col in base_ap_resid_cols:
            if col not in display_cols_ap_resid and col in ap_residuals_sorted_df.columns: # Add if not already added and exists
                 display_cols_ap_resid.append(col)
        
        final_display_cols_ap_resid = [col for col in display_cols_ap_resid if col in ap_residuals_sorted_df.columns]

        print("\nTop 5 Zip Codes with Apartment Prices Unusually High for Rents (Highest Positive Residuals):")
        print(ap_residuals_sorted_df[final_display_cols_ap_resid].head(5))

        print("\nTop 5 Zip Codes with Apartment Prices Unusually Low for Rents (Highest Negative Residuals):")
        print(ap_residuals_sorted_df[final_display_cols_ap_resid].tail(5))
    else:
        print("Not enough data points to calculate regression residuals for apartment prices vs. rents.")
else:
    print("merged_apartments_df_cleaned not found, is empty, or not a DataFrame. Please run the previous cells.")

# %% # Assess Best Transformation for Apartment Price vs. Rent Relationship
print("# %% # Assess Best Transformation for Apartment Price vs. Rent Relationship")

from scipy.stats import linregress # Ensure this import is run for the cell
import numpy as np # Ensure this is available for np.log; already imported at top but good for cell safety

if 'merged_apartments_plot_df' in locals() and isinstance(merged_apartments_plot_df, pd.DataFrame) and not merged_apartments_plot_df.empty:
    # Make a working copy for transformations
    trans_df = merged_apartments_plot_df.copy()

    # Ensure Price and Rent are positive for log transformations
    trans_df = trans_df[(trans_df['Apartment_Price'] > 0) & (trans_df['Rent'] > 0)]

    if not trans_df.empty and len(trans_df) >= 2: # Need at least 2 points for regression
        results = {}

        # 1. Linear-Linear: Price ~ Rent
        slope, intercept, r_value, p_value, std_err = linregress(trans_df['Rent'], trans_df['Apartment_Price'])
        results['Linear-Linear (Price vs Rent)'] = r_value**2

        # 2. Log-Linear (log Y): log(Price) ~ Rent
        slope_logy, intercept_logy, r_value_logy, p_value_logy, std_err_logy = linregress(trans_df['Rent'], np.log(trans_df['Apartment_Price']))
        results['Log-Linear (log(Price) vs Rent)'] = r_value_logy**2

        # 3. Linear-Log (log X): Price ~ log(Rent)
        slope_logx, intercept_logx, r_value_logx, p_value_logx, std_err_logx = linregress(np.log(trans_df['Rent']), trans_df['Apartment_Price'])
        results['Linear-Log (Price vs log(Rent))'] = r_value_logx**2

        # 4. Log-Log: log(Price) ~ log(Rent)
        slope_loglog, intercept_loglog, r_value_loglog, p_value_loglog, std_err_loglog = linregress(np.log(trans_df['Rent']), np.log(trans_df['Apartment_Price']))
        results['Log-Log (log(Price) vs log(Rent))'] = r_value_loglog**2

        print("R-squared values for different transformations of Apartment Price vs. Rent:")
        for model_name, r_squared in results.items():
            print(f"- {model_name}: {r_squared:.4f}")
        
        best_model = max(results, key=results.get)
        print(f"\nBased on R-squared, the best transformation appears to be: {best_model} (R-squared: {results[best_model]:.4f})")

    elif trans_df.empty or len(trans_df) < 2:
        print("Not enough valid (positive Price and Rent) data points to perform regression analysis after filtering.")
else:
    print("merged_apartments_plot_df not found, is empty, or not a DataFrame. Please run the previous cells.")

# %% # Linear Scatter Plot: Apartment Price vs. Rent (Rent <= $4k, Rent Share >= 60%)
print("# %% # Linear Scatter Plot: Apartment Price vs. Rent (Rent <= $4k, Rent Share >= 60%)")

if 'merged_apartments_df' in locals() and isinstance(merged_apartments_df, pd.DataFrame) and \
   'df_screened' in locals() and isinstance(df_screened, pd.DataFrame):

    # Start with a copy of apartment data
    plot_df_high_rent_share = merged_apartments_df.copy()

    # Ensure key columns are numeric for filtering and plotting
    plot_df_high_rent_share['Apartment_Price'] = pd.to_numeric(plot_df_high_rent_share['Apartment_Price'], errors='coerce')
    plot_df_high_rent_share['Rent'] = pd.to_numeric(plot_df_high_rent_share['Rent'], errors='coerce')
    
    # Apply rent cap
    rent_cap_filter = 4000
    plot_df_high_rent_share = plot_df_high_rent_share[plot_df_high_rent_share['Rent'] <= rent_cap_filter]
    
    # Drop NaNs from essential columns before merge attempt
    plot_df_high_rent_share = plot_df_high_rent_share.dropna(subset=['RegionName', 'Apartment_Price', 'Rent'])

    # Prepare for merge: Ensure RegionName_str for apartment data
    if 'RegionName_str' not in plot_df_high_rent_share.columns:
        plot_df_high_rent_share['RegionName_str'] = plot_df_high_rent_share['RegionName'].astype(str).str.zfill(5)
    
    # Merge with Census data (df_screened) to get rent_share and total_occ
    # df_screened should have 'zcta', 'rent_share', 'total_occ'
    if 'zcta' in df_screened.columns and 'rent_share' in df_screened.columns and 'total_occ' in df_screened.columns:
        plot_df_high_rent_share = pd.merge(
            plot_df_high_rent_share,
            df_screened[['zcta', 'rent_share', 'total_occ']],
            left_on='RegionName_str',
            right_on='zcta',
            how='inner'  # Use inner merge to keep only ZCTAs with census data
        )
        plot_df_high_rent_share['rent_share'] = pd.to_numeric(plot_df_high_rent_share['rent_share'], errors='coerce')
        plot_df_high_rent_share['total_occ'] = pd.to_numeric(plot_df_high_rent_share['total_occ'], errors='coerce')
    else:
        print("Warning: Necessary columns missing in df_screened. Cannot apply rent_share filter or use census data for aesthetics.")
        # To prevent errors later, create placeholder columns if merge failed but we want to proceed with just apartment data
        plot_df_high_rent_share['rent_share'] = np.nan 
        plot_df_high_rent_share['total_occ'] = np.nan

    # Apply rent_share filter
    min_rent_share_filter = 0.60
    plot_df_high_rent_share = plot_df_high_rent_share[plot_df_high_rent_share['rent_share'] >= min_rent_share_filter]

    # Drop NaNs from columns used for aesthetics or essential for plot, after all filtering and merging
    plot_df_high_rent_share = plot_df_high_rent_share.dropna(subset=['Apartment_Price', 'Rent', 'rent_share', 'total_occ'])
    plot_df_high_rent_share = plot_df_high_rent_share.replace([np.inf, -np.inf], np.nan).dropna(subset=['Apartment_Price', 'Rent', 'rent_share', 'total_occ'])

    if not plot_df_high_rent_share.empty and len(plot_df_high_rent_share) >=2:
        plt.figure(figsize=(14, 10))
        
        sns.scatterplot(
            data=plot_df_high_rent_share, 
            x='Rent', 
            y='Apartment_Price', 
            alpha=0.6,
            hue='rent_share',
            size='total_occ',
            sizes=(30, 400) 
        )
        
        sns.regplot(
            data=plot_df_high_rent_share, 
            x='Rent', 
            y='Apartment_Price', 
            scatter=False, 
            color='darkred',
            line_kws={'linestyle':'--'}
        )
        
        title_date_ap = latest_common_date_ap if 'latest_common_date_ap' in locals() else "Latest Month"
        plt.title(f'Linear Apartment Price vs. Rent (Rent <= ${rent_cap_filter:,}, Rent Share >= {min_rent_share_filter:.0%}, {title_date_ap})\nColored by Rent Share, Sized by Total Occupancy')
        plt.xlabel('Monthly Rent')
        plt.ylabel('Apartment Price')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend(title='Rent Share', loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    elif plot_df_high_rent_share.empty:
        print("No data available after applying all filters (Rent <= $4k, Rent Share >= 60%).")
    else: # Less than 2 points
        print("Not enough data points ( < 2) after applying filters to generate a meaningful plot.")
else:
    print("Required DataFrames (merged_apartments_df or df_screened) not found. Please ensure previous cells have run correctly.")

# %% # R-squared of Apartment Price vs. Rent by Total Occupancy Quintiles
print("# %% # R-squared of Apartment Price vs. Rent by Total Occupancy Quintiles")

from scipy.stats import linregress # Ensure this is available
import numpy as np # Ensure this is available

# We use the DataFrame prepared for the previous plot, which has the Rent <= $4k and Rent Share >= 60% filters
if 'plot_df_high_rent_share' in locals() and isinstance(plot_df_high_rent_share, pd.DataFrame) and not plot_df_high_rent_share.empty:
    df_for_quintile_analysis = plot_df_high_rent_share.copy()

    # Ensure Price, Rent, and total_occ are numeric and valid for regression/quintiles
    df_for_quintile_analysis['Apartment_Price'] = pd.to_numeric(df_for_quintile_analysis['Apartment_Price'], errors='coerce')
    df_for_quintile_analysis['Rent'] = pd.to_numeric(df_for_quintile_analysis['Rent'], errors='coerce')
    df_for_quintile_analysis['total_occ'] = pd.to_numeric(df_for_quintile_analysis['total_occ'], errors='coerce')

    # Remove rows where essential data for regression or quintiles is NaN, or Price/Rent are not positive
    df_for_quintile_analysis = df_for_quintile_analysis.dropna(subset=['Apartment_Price', 'Rent', 'total_occ'])
    df_for_quintile_analysis = df_for_quintile_analysis[(df_for_quintile_analysis['Apartment_Price'] > 0) & (df_for_quintile_analysis['Rent'] > 0)]

    if not df_for_quintile_analysis.empty and df_for_quintile_analysis['total_occ'].nunique() >= 5 and len(df_for_quintile_analysis) >= 10: # Need enough data for meaningful quintiles & regressions
        try:
            df_for_quintile_analysis['Total Occupancy Quintile'] = pd.qcut(df_for_quintile_analysis['total_occ'], 5, labels=[f"Quintile {i+1}" for i in range(5)])
            print("R-squared for Apartment Price vs. Rent, by Total Occupancy Quintile (Rent <= $4k, Rent Share >= 60%):")
            
            results_by_quintile = {}
            for quintile_name in df_for_quintile_analysis['Total Occupancy Quintile'].cat.categories:
                quintile_data = df_for_quintile_analysis[df_for_quintile_analysis['Total Occupancy Quintile'] == quintile_name]
                
                if len(quintile_data) >= 2: # Need at least 2 points for regression
                    slope, intercept, r_value, p_value, std_err = linregress(quintile_data['Rent'], quintile_data['Apartment_Price'])
                    r_squared = r_value**2
                    results_by_quintile[quintile_name] = r_squared
                    print(f"- {quintile_name} (N={len(quintile_data)}): R-squared = {r_squared:.4f}")
                else:
                    print(f"- {quintile_name}: Not enough data points (N={len(quintile_data)}) for regression.")
                    results_by_quintile[quintile_name] = np.nan
        except ValueError as e:
            print(f"Error creating quintiles for total_occ: {e}. Ensure enough unique values and data points.")
        except Exception as e:
            print(f"An unexpected error occurred during quintile analysis: {e}")

    elif df_for_quintile_analysis.empty:
        print("No valid data points available for quintile analysis after filtering.")
    else:
        print("Not enough unique 'total_occ' values or overall data points to create 5 meaningful quintiles and perform regressions.")
else:
    print("'plot_df_high_rent_share' (data filtered for Rent <= $4k and Rent Share >= 60%) not found or is empty. Please ensure the previous cell ran successfully.")

# %% # Linear Scatter Plot: Apt Price vs. Rent (Rent <= $4k, Rent Share >= 60%, Total Occ Quintiles 2-4)
print("# %% # Linear Scatter Plot: Apt Price vs. Rent (Rent <= $4k, Rent Share >= 60%, Total Occ Quintiles 2-4)")

from scipy.stats import linregress # Should be available, but for cell safety
import numpy as np # Should be available, but for cell safety

# Use the DataFrame prepared earlier with Rent <= $4k and Rent Share >= 60% filters
if 'plot_df_high_rent_share' in locals() and isinstance(plot_df_high_rent_share, pd.DataFrame) and not plot_df_high_rent_share.empty:
    df_quintiles_2_3_4 = plot_df_high_rent_share.copy()

    # Ensure relevant columns are numeric and clean for quintiles and plotting
    df_quintiles_2_3_4['Apartment_Price'] = pd.to_numeric(df_quintiles_2_3_4['Apartment_Price'], errors='coerce')
    df_quintiles_2_3_4['Rent'] = pd.to_numeric(df_quintiles_2_3_4['Rent'], errors='coerce')
    df_quintiles_2_3_4['total_occ'] = pd.to_numeric(df_quintiles_2_3_4['total_occ'], errors='coerce')
    df_quintiles_2_3_4['rent_share'] = pd.to_numeric(df_quintiles_2_3_4['rent_share'], errors='coerce') # For hue

    # Drop NaNs from essential columns for quintile calculation and plotting
    df_quintiles_2_3_4 = df_quintiles_2_3_4.dropna(subset=['Apartment_Price', 'Rent', 'total_occ', 'rent_share'])
    df_quintiles_2_3_4 = df_quintiles_2_3_4[(df_quintiles_2_3_4['Apartment_Price'] > 0) & (df_quintiles_2_3_4['Rent'] > 0)]

    # Calculate Total Occupancy Quintiles
    quintile_labels = [f"Quintile {i+1}" for i in range(5)]
    if not df_quintiles_2_3_4.empty and df_quintiles_2_3_4['total_occ'].nunique() >= 5:
        try:
            df_quintiles_2_3_4['Total Occupancy Quintile'] = pd.qcut(df_quintiles_2_3_4['total_occ'], 5, labels=quintile_labels)
            
            # Filter for Quintiles 2, 3, and 4
            selected_quintiles = ['Quintile 2', 'Quintile 3', 'Quintile 4']
            df_plot_final_subset = df_quintiles_2_3_4[df_quintiles_2_3_4['Total Occupancy Quintile'].isin(selected_quintiles)]

            if not df_plot_final_subset.empty and len(df_plot_final_subset) >= 2:
                plt.figure(figsize=(14, 10))
                
                sns.scatterplot(
                    data=df_plot_final_subset, 
                    x='Rent', 
                    y='Apartment_Price', 
                    alpha=0.7, # Slightly increased alpha for visibility
                    hue='rent_share',
                    size='total_occ',
                    sizes=(40, 450) # Adjusted sizes for emphasis
                )
                
                sns.regplot(
                    data=df_plot_final_subset, 
                    x='Rent', 
                    y='Apartment_Price', 
                    scatter=False, 
                    color='blue', # Changed color for distinction
                    line_kws={'linestyle':'-', 'linewidth': 2} # Solid, thicker line
                )
                
                rent_cap_filter = 4000 # from previous cell context
                min_rent_share_filter = 0.60 # from previous cell context
                title_date_ap = latest_common_date_ap if 'latest_common_date_ap' in locals() else "Latest Month"
                
                plt.title(f'Linear Apt Price vs. Rent (Total Occ Quintiles 2-4)\n(Rent <= ${rent_cap_filter:,}, Rent Share >= {min_rent_share_filter:.0%}, {title_date_ap})\nColored by Rent Share, Sized by Total Occupancy')
                plt.xlabel('Monthly Rent')
                plt.ylabel('Apartment Price')
                plt.grid(True, which="both", ls="-", alpha=0.4)
                plt.legend(title='Rent Share', loc='upper left', bbox_to_anchor=(1.02, 1))
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.show()
                
                # Optional: R-squared for this specific subset
                slope, intercept, r_value, p_value, std_err = linregress(df_plot_final_subset['Rent'], df_plot_final_subset['Apartment_Price'])
                print(f"\nR-squared for Apartment Price vs. Rent (Total Occ. Quintiles 2-4, Rent <= $4k, Rent Share >= 60%): {r_value**2:.4f}")

            elif df_plot_final_subset.empty:
                print("No data available after filtering for Total Occupancy Quintiles 2, 3, and 4.")
            else: # Less than 2 points in the subset
                print("Not enough data points ( < 2) in Total Occupancy Quintiles 2, 3, and 4 to generate a plot.")

        except ValueError as e:
            print(f"Error creating total_occ quintiles for plotting: {e}. Ensure enough unique values.")
    elif df_quintiles_2_3_4.empty:
        print("No valid data points available for plotting after initial cleaning (Price/Rent > 0).")
    else: # Not enough unique total_occ values for 5 quintiles
        print("Not enough unique 'total_occ' values to create 5 quintiles for this specific plot.")
else:
    print("'plot_df_high_rent_share' (data filtered for Rent <= $4k and Rent Share >= 60%) not found or is empty. Please ensure the previous cell ran successfully.")

# %%
