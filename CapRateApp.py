#%% Load libraries
print("#%% Load libraries")
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import datetime
import os # Import os module

#%% Load Zillow buy and rent data
print("#%% Load Zillow buy and rent data")
buy_file_path = "Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month_SFH.csv"
rent_file_path = "Metro_zori_uc_sfr_sm_month_SFR.csv"

print(f"Checking buy file: {buy_file_path}")
if os.path.exists(buy_file_path):
    print(f"Buy file size: {os.path.getsize(buy_file_path)} bytes")
else:
    print(f"Buy file NOT FOUND at {buy_file_path}")

print(f"Checking rent file: {rent_file_path}")
if os.path.exists(rent_file_path):
    print(f"Rent file size: {os.path.getsize(rent_file_path)} bytes")
else:
    print(f"Rent file NOT FOUND at {rent_file_path}")

buy = pd.read_csv(buy_file_path, encoding='latin-1')
rent = pd.read_csv(rent_file_path, encoding='latin-1')

#%% Load crosswalk file
print("#%% Load crosswalk file")
crosswalk = pd.read_csv("CountyCrossWalk_Zillow.csv", encoding='latin-1')

#%% Check crosswalk columns
print("#%% Check crosswalk columns")
print(crosswalk.columns)

#%% Get Census MSA population data
print("#%% Get Census MSA population data")
census_url = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/metro/totals/cbsa-est2023-alldata.csv"
census = pd.read_csv(census_url, encoding='latin-1')

#%% Prepare Census population data
print("#%% Prepare Census population data")
census = census.rename(columns={'CBSA': 'cbsa_code', 'POPESTIMATE2023': 'population'})
census['cbsa_code'] = census['cbsa_code'].astype(str).str.zfill(5)
census = census[['cbsa_code', 'population']]

#%% Prepare crosswalk for merging
print("#%% Prepare crosswalk for merging")
crosswalk = crosswalk.rename(columns={'MetroRegionID_Zillow': 'RegionID'})
crosswalk['cbsa_code'] = crosswalk['CBSACode'].astype(str).str.zfill(5)
crosswalk = crosswalk[['RegionID', 'cbsa_code']].drop_duplicates()

#%% Merge buy and rent data
print("#%% Merge buy and rent data")
buy_id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
rent_id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
buy_long = buy.melt(id_vars=buy_id_vars, var_name='date', value_name='price')
rent_long = rent.melt(id_vars=rent_id_vars, var_name='date', value_name='rent')
df = pd.merge(buy_long, rent_long, on=buy_id_vars + ['date'], how='inner')

#%% Merge with crosswalk and population
print("#%% Merge with crosswalk and population")
df = pd.merge(df, crosswalk, on='RegionID', how='left')
df = pd.merge(df, census, on='cbsa_code', how='left')

#%% Convert date column and determine latest reliable date
print("#%% Convert date column and determine latest reliable date")
df['date'] = pd.to_datetime(df['date'])

# Create a temporary df for calculating latest_date, ensuring it's from rows with actual positive data
df_for_date_calc = df[
    (df['price'].notnull()) & (df['price'] > 0) &  # Ensure positive price
    (df['rent'].notnull()) & (df['rent'] > 0) &    # Ensure positive rent
    (df['RegionType'] == 'msa')
].copy()

if df_for_date_calc.empty:
    print("Warning: No data available after filtering for positive price/rent and MSA to determine latest_date.")
    latest_date = None
    df_latest = pd.DataFrame() 
else:
    latest_date = df_for_date_calc['date'].max()
    
    # Now filter the original df using this reliable latest_date and ensuring positive values
    df_latest = df[
        (df['date'] == latest_date) & 
        (df['RegionType'] == 'msa') &
        (df['population'].notnull()) &
        (df['price'].notnull()) & (df['price'] > 0) & # Ensure positive price
        (df['rent'].notnull()) & (df['rent'] > 0)     # Ensure positive rent
    ].copy()

    # It's good practice to check if df_latest became empty after all filters
    # if df_latest.empty:
    #     print(f"Warning: df_latest is empty for date {latest_date} after all filters including population and positive price/rent.")

#%% Calculate cap rate
print("#%% Calculate cap rate")
# Ensure df_latest is not empty before trying to calculate cap_rate
if not df_latest.empty:
    df_latest['cap_rate'] = df_latest['rent'] * 12 / df_latest['price']
else:
    # If df_latest is empty, create an empty 'cap_rate' column to prevent errors later
    df_latest['cap_rate'] = pd.Series(dtype='float64')

#%% Streamlit UI: Title and dropdown
print("#%% Streamlit UI: Title and dropdown")
VERSION_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.set_page_config(page_title="Cap Rate Monitor", layout="wide")
st.caption(f"Version: {VERSION_TIMESTAMP}")
st.title("Cap Rate Monitor")
city = st.selectbox("Highlight a city:", sorted(df_latest['RegionName'].unique()))

#%% Prepare colors and plot data
print("#%% Prepare colors and plot data")
highlight_color = "#0BB4FF"
other_color = "#DADFCE"
df_latest['color'] = df_latest['RegionName'].apply(lambda x: highlight_color if x == city else other_color)

#%% Plotly scatter plot
print("#%% Plotly scatter plot")
fig = px.scatter(
    df_latest,
    x="price",
    y="rent",
    size="population",
    color="color",
    hover_name="RegionName",
    size_max=60,
    color_discrete_map="identity",
    labels={"price": "Median Price", "rent": "Median Rent"},
    title=f"Price vs Rent (Bubble size = Population) — {latest_date}"
)
fig.update_traces(marker=dict(line=dict(width=1, color="#3D3733")))
fig.update_layout(showlegend=False, plot_bgcolor="#F6F7F3", paper_bgcolor="#F6F7F3")

#%% Show plot
print("#%% Show plot")
st.plotly_chart(fig, use_container_width=True)