#%% Load libraries
print("#%% Load libraries")
import pandas as pd
import streamlit as st
import plotly.express as px
import requests

#%% Load Zillow buy and rent data
print("#%% Load Zillow buy and rent data")
buy = pd.read_csv("Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month_SFH.csv", encoding='latin-1')
rent = pd.read_csv("Metro_zori_uc_sfr_sm_month_SFR.csv", encoding='latin-1')

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
buy_long = buy.melt(id_vars=['RegionID', 'RegionName'], var_name='date', value_name='price')
rent_long = rent.melt(id_vars=['RegionID', 'RegionName'], var_name='date', value_name='rent')
df = pd.merge(buy_long, rent_long, on=['RegionID', 'RegionName', 'date'], how='inner')

#%% Merge with crosswalk and population
print("#%% Merge with crosswalk and population")
df = pd.merge(df, crosswalk, on='RegionID', how='left')
df = pd.merge(df, census, on='cbsa_code', how='left')

#%% Filter for latest month and MSAs with population data
print("#%% Filter for latest month and MSAs with population data")
latest_date = df['date'].max()
df_latest = df[(df['date'] == latest_date) & (~df['population'].isna())]

#%% Calculate cap rate
print("#%% Calculate cap rate")
df_latest['cap_rate'] = df_latest['rent'] * 12 / df_latest['price']

#%% Streamlit UI: Title and dropdown
print("#%% Streamlit UI: Title and dropdown")
st.set_page_config(page_title="Cap Rate Monitor", layout="wide")
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