import streamlit as st
import matplotlib.pyplot as plt
from pycirclize import Circos
import tempfile
import pandas as pd
import altair as alt
import os

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
correspondance = correspondance.merge(countries[['DE_code','Region']]).drop(columns='DE_code')
correspondance['dotstat_code']=correspondance['dotstat_code'].astype(int)
# df_2021 = data[data.Year == 2021].dropna(subset=['Value'])
df_2021 = df_2021.merge(correspondance.rename(columns={'dotstat_code':'RECIPIENT','Region':'Recipient_Region'}), how='left')
df_2021 = df_2021.merge(correspondance.rename(columns={'dotstat_code':'DONOR','Region':'Donor_Region'}), how='left')


regions = ['Europe & Central Asia', 'Middle East & North Africa', 'Sub-Saharan Africa',
 'Latin America & Caribbean', 'South Asia','North America', 'East Asia & Pacific']

# Sample code for processing the data (you already have this part in your script)
df_regions = df_2021[
    (df_2021['Recipient_Region'].isin(regions)) &
    (df_2021['Donor_Region'].isin(regions)) &
    (df_2021['Aid type'] == 'ODA: Total Net')
]

df_regions['Value_k'] = df_regions['Value'] / 1000  # Dividing by 1000 for easier plotting

# Group flows
grouped = df_regions.groupby(['Donor_Region', 'Recipient_Region'], as_index=False)['Value_k'].sum()
flow_matrix = grouped.pivot(index='Donor_Region', columns='Recipient_Region', values='Value_k').fillna(0)

# Define colors
colors = {
    'Europe & Central Asia': '#5778a4',
    'Middle East & North Africa': '#ff7f00',
    'Sub-Saharan Africa': '#ffd60a',
    'Latin America & Caribbean': '#e78ac3',
    'South Asia': '#f94144',
    'East Asia & Pacific': '#7400b8',
    'North America': '#06d6a0',
}

# Create the Circos plot
circos = Circos.chord_diagram(
    flow_matrix,
    # cmap="viridis",  # Use a colormap suitable for financial flows
    cmap=colors,
    space=2,
    ticks_interval=10,
    label_kws=dict(
        size=12,
        r=110,
        ),
    link_kws=dict(
        ec="black",
        lw=0.3,
        direction=1,
        alpha=0.7,
        # ribbon=True
    ),
    ticks_kws=dict(
        label_orientation="horizontal"
    )
)

# Add title to the Circos plot
circos.text("ODA Financial Flows Between Regions", deg=0, r=150, size=14)
fig = circos.plotfig()


poverty =poverty.merge(countries[['DE_code','Region','Income Group']].rename(columns={"DE_code":'Country Code'}), how='left', on='Country Code')
poverty['Total Population (in million)'] = poverty['Population, total'] / 1_000_000
poverty['Number of Poor (in million)'] = (poverty['Poverty headcount ratio at $2.15 a day'] / 100) * poverty['Total Population (in million)']

## Global overview world population and extreme poor
df_world = poverty[(poverty['Country Name'] == 'World')]
poor = alt.Chart(df_world[(df_world.Year >= 1990) & (df_world.Year <= 2022)]).mark_area(opacity=0.7).encode(
    x='Year:O',
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of People')),
    color=alt.value('red')
).properties(
    title='Number of People Living in Extreme Poverty',
    width=600,
    height=300
)
popTotal = alt.Chart(df_world[(df_world.Year >= 1990) & (df_world.Year <= 2022)]).mark_area(opacity=0.5).encode(
    x='Year:O',
    y=alt.Y('Total Population (in million):Q', axis=alt.Axis(title='Millions of People')), #, scale=alt.Scale(domain=(0, 8000000))),
    color=alt.value('gray')
).properties(
    title='Total Population in the World',
    width=600,
    height=300
)
chart1 = popTotal + poor

### Chart 2: Zoom in regional poverty
regions = [
    'South Asia', 'Europe & Central Asia', 'Middle East & North Africa',
    'East Asia & Pacific', 'Sub-Saharan Africa', 'Latin America & Caribbean'
]

df_pov_regions = poverty[poverty['Country Name'].isin(regions)].drop(columns='Region')
df_pov_regions = df_pov_regions.rename(columns={'Country Name': 'Region'})
temp = df_pov_regions.sort_values(by=['Region', 'Year'])
temp = temp.groupby('Region').ffill()

df_pov_regions = temp.merge(df_pov_regions[['Region']], left_index=True, right_index=True)
chart2 = alt.Chart(df_pov_regions[(df_pov_regions.Year >= 1990) & (df_pov_regions.Year <= 2022)]).mark_area(opacity=0.7).encode(
    x='Year:O',
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of People')),
    color=alt.Color('Region:N')
).properties(
    title='Number of People Living in Extreme Poverty',
    width=400,
    height=400
)



st.set_page_config(layout="centered")

# Title
st.title("Global Poverty & Financial Aid: A Story of Progress and Challenges")

# Introduction
st.markdown("""
Welcome to this interactive dashboard exploring the journey of global poverty reduction, 
regional disparities, and the role of financial aid. 
Through data and visualizations, we will uncover how the world has made significant strides 
— and where challenges remain.
""")

# Section 1: Global Overview
st.header("🌍 Global Overview: Population Growth & Poverty Reduction")

st.markdown("""
Over the last few decades, the world has seen unprecedented population growth. 
However, alongside this, the share of people living in **extreme poverty** has dramatically decreased.
""")

# Placeholder for population growth vs. poverty rate chart
# (You will need to load your data here)
st.subheader("Population Growth vs. Extreme Poverty Rate")
st.altair_chart(chart1, use_container_width=True)


# Section 2: Regional Focus
st.header("🗺️ Regional Focus: Progress with Disparities")

st.markdown("""
While global averages tell one story, a closer look at different regions reveals more.
**South Asia and East Asia** have led the way in poverty reduction, 
driven by rapid economic transformation.
Other regions, particularly Sub-Saharan Africa, still face significant challenges.
""")

# Placeholder for regional comparison chart
st.subheader("Regional Poverty Rates Over Time")
st.altair_chart(chart2, use_container_width=True)
# fig, ax = plt.subplots()
# ax.plot(...)
# st.pyplot(fig)

# Section 3: Income Growth & Economic Transformation
st.header("💰 Growth in Income: The Engine of Poverty Reduction")

st.markdown("""
A major driver behind poverty reduction has been **economic growth**, especially in populous countries like **China** and **India**.
Understanding changes in **GDP composition** over time helps explain how these economies shifted from agriculture to industry and services.
""")

st.subheader("GDP Growth")
# fig, ax = plt.subplots()
# ax.plot(...)
# st.pyplot(fig)

st.subheader("GDP Composition Comparison : India / China (1990 vs. 2010)")
# fig, ax = plt.subplots()
# ax.bar(...)
# st.pyplot(fig)

# Section 4: Impact of ODA & Aid Flows
st.header("🌐 The Role of Official Development Assistance (ODA)")

st.markdown("""
Beyond domestic growth, **international aid** has played a critical role, 
particularly in supporting low-income countries.
Tracking **ODA flows** helps us understand where aid is directed and its potential impact.
""")

# Placeholder for ODA/Aid flows chart
st.subheader("Aid Flows Over Time")

fig = circos.plotfig()
st.pyplot(fig)
# fig, ax = plt.subplots()
# ax.plot(...)
# st.pyplot(fig)

# Conclusion
st.header("📊 Conclusion: Lessons & Ongoing Challenges")

st.markdown("""
The global fight against poverty has achieved remarkable progress, 
but it remains uneven. Economic growth, targeted aid, and policy choices 
continue to shape the future. Stay curious, explore the data, 
and let's keep the momentum going!
""")

