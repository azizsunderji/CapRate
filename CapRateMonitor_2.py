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
import requests
import plotly.graph_objects as go
import plotly.io as pio

# Set rcParams to make sure text is saved as editable text in SVG files
plt.rcParams['svg.fonttype'] = 'none'  # Makes text in SVGs editable
plt.rcParams['font.sans-serif'] = 'Arial'  # (Optional) Set a default font like Arial

# set display to show max columns
pd.set_option('display.max_columns', None)

# set display to show 300 rows maximum
pd.set_option('display.max_rows', 50)

# %% [markdown]
# ## Set Directory Paths

# %%
# Set main directory (current directory)
main_dir = os.getcwd()
print(f"Main directory: {main_dir}")

# Set reference directory
reference_dir = os.path.join(os.path.expanduser('~'), 'Home Economics Dropbox', 'Aziz Sunderji', 'Home Economics', 'Reference', 'Shapefiles')
print(f"Reference directory: {reference_dir}")

# %% [markdown]
# ## Load Zillow Data

# %% Load Zillow SFH (buy) and SFR (rent) data
buy_df = pd.read_csv('Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month_SFH.csv', encoding='latin-1')
rent_df = pd.read_csv('Metro_zori_uc_sfr_sm_month_SFR.csv', encoding='latin-1')

print("Buy data shape:", buy_df.shape)
print("Rent data shape:", rent_df.shape)

# %% Examine the structure of both dataframes
print("\nBuy DataFrame Info:")
print(buy_df.info())
print("\nBuy DataFrame Head:")
print(buy_df.head())

print("\nRent DataFrame Info:")
print(rent_df.info())
print("\nRent DataFrame Head:")
print(rent_df.head())

# %% Inspect Zillow RegionID values and types
print('First 10 RegionID values from Zillow data:')
print(buy_df['RegionID'].head(10))
print('\nRegionID dtype:', buy_df['RegionID'].dtype)
print('\nUnique RegionID count:', buy_df['RegionID'].nunique())

# %% Merge buy and rent dataframes
# First, melt both dataframes to get them in long format
buy_melted = pd.melt(buy_df, 
                     id_vars=['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName'],
                     var_name='Date',
                     value_name='Price')

rent_melted = pd.melt(rent_df,
                      id_vars=['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName'],
                      var_name='Date',
                      value_name='Rent')

# Merge the melted dataframes
merged_df = pd.merge(buy_melted, rent_melted, 
                    on=['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'Date'],
                    how='inner')

# Convert Date to datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Sort by RegionName and Date
merged_df = merged_df.sort_values(['RegionName', 'Date'])

print("\nMerged DataFrame Info:")
print(merged_df.info())
print("\nMerged DataFrame Head:")
print(merged_df.head())

# %% Create scatter plot of price vs rent for most recent month
# Get the most recent date
latest_date = merged_df['Date'].max()

# Filter for the most recent date and only MSA regions
latest_data = merged_df[
    (merged_df['Date'] == latest_date) & 
    (merged_df['RegionType'] == 'msa')
].copy()

# %% Download and inspect raw MSA population data from Census Bureau API
import requests
import pandas as pd

base_url = "https://api.census.gov/data/2020/dec/pl"
params = {
    'get': 'P1_001N',  # Total population
    'for': 'metropolitan statistical area/micropolitan statistical area:*',
    'key': '06048dc3bd32068702b5ef9b49875ec0c5ca56ce'
}

response = requests.get(base_url, params=params)
if response.status_code == 200:
    data = response.json()
    msa_pop_data = pd.DataFrame(data[1:], columns=data[0])
    print('Census MSA population data (first 10 rows):')
    print(msa_pop_data.head(10))
    print('\nCensus columns:', msa_pop_data.columns.tolist())
else:
    print('Failed to fetch population data from Census Bureau API')

# %% Load the crosswalk file to map RegionID to CBSA code
crosswalk_df = pd.read_csv('CountyCrossWalk_Zillow.csv', encoding='latin-1')
print('Crosswalk columns:', crosswalk_df.columns.tolist())
print(crosswalk_df.head())

# %% Merge crosswalk with Zillow data to add CBSA code
# We'll use MetroRegionID_Zillow and CBSACode from the crosswalk
zillow_with_cbsa = buy_df.merge(
    crosswalk_df[['MetroRegionID_Zillow', 'CBSACode']].drop_duplicates(),
    left_on='RegionID', right_on='MetroRegionID_Zillow', how='left')
print('Zillow data with CBSA code (head):')
print(zillow_with_cbsa[['RegionID', 'RegionName', 'CBSACode']].head())

# %% Ensure CBSA code is the same type and format in both dataframes (zero-padded 5-digit string)
zillow_with_cbsa['CBSACode'] = zillow_with_cbsa['CBSACode'].apply(
    lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) and str(x).lower() != 'nan' else None
)
msa_pop_data['metropolitan statistical area/micropolitan statistical area'] = msa_pop_data['metropolitan statistical area/micropolitan statistical area'].str.strip()
msa_pop_data['CBSA'] = msa_pop_data['metropolitan statistical area/micropolitan statistical area'].apply(lambda x: str(int(x)).zfill(5) if pd.notnull(x) else None)

# %% Merge population data onto Zillow data using CBSA code
zillow_with_pop = zillow_with_cbsa.merge(
    msa_pop_data[['CBSA', 'P1_001N']],
    left_on='CBSACode', right_on='CBSA', how='left')

# %% Show the head of the resulting dataframe with population info
print('Zillow data with population info (head):')
print(zillow_with_pop[['RegionID', 'RegionName', 'CBSACode', 'P1_001N']].head())

# %% Interactive scatter plot with dropdown to highlight selected city
import plotly.graph_objects as go

# Prepare data for the latest month, MSA regions, and non-null population, price, and rent
df_plot = merged_df.merge(
    zillow_with_pop[['RegionID', 'CBSACode', 'P1_001N']],
    on='RegionID', how='left'
)
latest_date = df_plot['Date'].max()
df_latest = df_plot[(df_plot['Date'] == latest_date) &
                    (df_plot['RegionType'] == 'msa') &
                    (df_plot['P1_001N'].notnull()) &
                    (df_plot['Price'].notnull()) &
                    (df_plot['Rent'].notnull())].copy()

# Color palette
highlight_color = '#0BB4FF'  # Blue
other_color = '#D3D3D3'       # Light gray

# Scale bubble sizes more naturally
marker_sizes = (df_latest['P1_001N'].astype(float) / 100000).clip(3, 30)
marker_colors = [highlight_color] * len(df_latest)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_latest['Rent'],
    y=df_latest['Price'],
    mode='markers',
    marker=dict(
        size=marker_sizes,
        color=marker_colors,
        opacity=0.7,
        line=dict(width=1, color='DarkSlateGrey')
    ),
    text=df_latest['RegionName'],
    customdata=df_latest[['RegionName', 'P1_001N', 'Price', 'Rent', 'CBSACode']],
    hovertemplate='<b>%{customdata[0]}</b><br>Population: %{customdata[1]}<br>Price: %{customdata[2]:,.0f}<br>Rent: %{customdata[3]:,.0f}<br>CBSA: %{customdata[4]}<extra></extra>'
))

# Dropdown menu for city selection
buttons = []
for i, city in enumerate(df_latest['RegionName']):
    colors = [other_color] * len(df_latest)
    opacities = [0.3] * len(df_latest)
    colors[i] = highlight_color
    opacities[i] = 1.0
    buttons.append(dict(
        method='restyle',
        label=city,
        args=[
            {'marker.color': [colors],
             'marker.opacity': [opacities]}
        ]
    ))

# Add a button to reset to all blue
buttons.insert(0, dict(
    method='restyle',
    label='Show All',
    args=[
        {'marker.color': [[highlight_color]*len(df_latest)],
         'marker.opacity': [[0.7]*len(df_latest)]}
    ]
))

fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction='down',
        showactive=True,
        x=0.99,
        xanchor='right',
        y=1.13,
        yanchor='top',
        pad={'r': 2, 't': 2},
        font=dict(size=11)
    )],
    title=f'Purchase Price vs Monthly Rent by MSA ({latest_date.strftime("%B %Y")})<br>Bubble size = Population',
    xaxis_title='Monthly Rent ($)',
    yaxis_title='Purchase Price ($)',
    template='plotly_white',
    width=1000,
    height=700,
    xaxis=dict(rangemode='tozero'),
    yaxis=dict(rangemode='tozero')
)

# Explicitly set the renderer to browser
pio.renderers.default = "browser"

# Add these lines for debugging nbformat
print("Attempting to import nbformat for Plotly...")
try:
    import nbformat
    print(f"Successfully imported nbformat version: {nbformat.__version__}")
    print(f"nbformat path: {nbformat.__file__}")
except ImportError as e:
    print(f"Failed to import nbformat: {e}")
    print("Please ensure nbformat is installed in the correct environment.")

fig.show()
