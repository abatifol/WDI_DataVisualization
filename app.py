import streamlit as st
import matplotlib.pyplot as plt
from pycirclize import Circos
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import pandas as pd
import altair as alt
import os
import streamlit_shadcn_ui as ui

alt.data_transformers.enable("vegafusion")

# Set Streamlit page config
st.set_page_config(
    layout="wide",
    page_title="Global Poverty and Financial Flows Dashboard",
    page_icon="🌍"
)

#################################################
## ---------------- Load data ---------------- ##
#################################################

@st.cache_data
def load_data():
    df_2021 = pd.read_csv('./data/oda_2021.csv')
    # df_2021 = data[data.Year == 2021].dropna(subset=['Value'])
    print('ODA data loaded')
    countries = pd.read_csv('./data/WDICountry.csv').rename(columns={"Country Code":"DE_code"})
    print("countries data loaded")
    correspondance = pd.read_csv('./data/Correspondences_DAC2a.csv')
    print('correspondance data loaded')
    poverty = pd.read_csv('data/oda_poverty_world.csv')
    return df_2021, countries, correspondance, poverty

df_2021, countries, correspondance, poverty = load_data()

#################################################
## --------------- Intro charts -------------- ##
#################################################
# Prepare poverty data
poverty = poverty.merge(
    countries[['DE_code', 'Region', 'Income Group']].rename(columns={"DE_code": 'Country Code'}),
    how='left',
    on='Country Code'
)
poverty['Total Population (in million)'] = poverty['Population, total'] / 1_000_000
poverty['Number of Poor (in million)'] = (poverty['Poverty headcount ratio at $2.15 a day'] / 100) * poverty['Total Population (in million)']

# Filter world data for global view
df_world = poverty[poverty['Country Name'] == 'World']

# --- Chart 1: Global Population and Poverty ---
poor = alt.Chart(df_world[(df_world.Year >= 1990) & (df_world.Year <= 2022)]).mark_area(opacity=0.7).encode(
    x='Year:O',
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of People')),
    color=alt.value('red')
).properties(
    title='Number of People Living in Extreme Poverty',
)
popTotal = alt.Chart(df_world[(df_world.Year >= 1990) & (df_world.Year <= 2022)]).mark_area(opacity=0.5).encode(
    x='Year:O',
    y=alt.Y('Total Population (in million):Q', axis=alt.Axis(title='Millions of People')), #, scale=alt.Scale(domain=(0, 8000000))),
    color=alt.value('gray')
).properties(
    title='Total Population in the World',
)
# Create manual legend using text marks
legend_text = alt.Chart(pd.DataFrame({
    'label': ['Global Population', 'Extreme Poor'],
    'color': ['gray', 'red'],
    'y': [7_500, 6_800]  # Adjust positions based on your data scale
})).mark_text(align='left', dx=5).encode(
    x=alt.value(10),  # Position at the left
    y='y:Q',
    text='label:N',
    color=alt.Color('color:N', scale=None)
)

legend_points = alt.Chart(pd.DataFrame({
    'color': ['gray', 'red'],
    'y': [7_500, 6_800]
})).mark_point(filled=True, size=100).encode(
    x=alt.value(0),
    y='y:Q',
    color=alt.Color('color:N', scale=None)
)

chart1 = (popTotal + poor + legend_points + legend_text).properties(
    title=alt.TitleParams(
        text='World Population Growth and Decline in Extreme Poverty (1990–2022)', 
        anchor='middle', 
        fontSize=16,  
        fontWeight='bold'  
    ),
    width=500,
    height=400
)


# --- Chart 2: Regional Breakdown of Extreme Poverty ---
regions = [
    'South Asia', 'Europe & Central Asia', 'Middle East & North Africa',
    'East Asia & Pacific', 'Sub-Saharan Africa', 'Latin America & Caribbean'
]

df_pov_regions = poverty[poverty['Country Name'].isin(regions)].drop(columns='Region')
df_pov_regions = df_pov_regions.rename(columns={'Country Name': 'Region'})
temp = df_pov_regions.sort_values(by=['Region', 'Year'])
# fill the missing years with the last known value
temp = temp.groupby('Region').ffill()
df_pov_regions = temp.merge(df_pov_regions[['Region']], left_index=True, right_index=True)

chart2 = alt.Chart(df_pov_regions[(df_pov_regions.Year >= 1990) & (df_pov_regions.Year <= 2022)]).mark_area(opacity=0.7).encode(
    x='Year:O',
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of People')),
    color=alt.Color(
        'Region:N',
        scale=alt.Scale(scheme='turbo'),
        legend=alt.Legend(
        title="Region",
        orient='top-right'))
).properties(
    title=alt.TitleParams(
        text='Regional Trends in Extreme Poverty (1990-2022)',  
        anchor='middle',  
        fontSize=16,  
        fontWeight='bold'  
    ),
    width=500,
    height=500
)

#################################################
## ------------ Streamlit Display ------------ ##
#################################################

st.title("🌍 How economic growth and financial aid shape the fight against poverty?")

# --- Part 1: Overview of Global Poverty --- #
st.subheader("Part 1: Global Overview: Population Growth & Poverty Reduction")

# Add top-level indicators
latest_year = 2022
latest_population = df_world[df_world['Year'] == latest_year]['Total Population (in million)'].values[0]
latest_poverty = df_world[df_world['Year'] == latest_year]['Number of Poor (in million)'].values[0]
poverty_rate = df_world[df_world['Year'] == latest_year]['Poverty headcount ratio at $2.15 a day'].values[0]

st.markdown(f"### 🌐 Latest Snapshot: {latest_year}")
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    ui.metric_card(title="Extreme Poverty Rate (under $2.15/day)", content=f"{poverty_rate:.2f}%", description="37,7% in 1990" )
with kpi2:
    ui.metric_card(title="Global Population (Million)", content=f"{latest_population:.0f}", description="+2.7 billion since 1990")
with kpi3:
    ui.metric_card(title="People in Extreme Poverty (Million)", content=f"{latest_poverty:,.0f}", description="-1.3 billion since 1990")


st.markdown("""
In the last three decades, the global population has grown by over 2.5 billion people, while the number of people living in extreme poverty has significantly decreased.
However, this positive global trend hides regional disparities. **South Asia and East Asia** have led the way in poverty reduction, driven by rapid economic transformation.
while Sub-Saharan Africa still faces significant challenges.
""")
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(chart1, use_container_width=True)
    st.markdown("**Source:** World Bank,World Development indicator Database (WDI)")
with col2:
    st.altair_chart(chart2, use_container_width=True)

st.markdown("**Note:** The number of extreme poor is calculated using the poverty headcount ratio at $2.15 a day (2017 PPP) and the total population.")

# --- Part 2: GDP and Poverty Reduction --- #

# --- Part 3: Financial Flows and Poverty Reduction --- #
