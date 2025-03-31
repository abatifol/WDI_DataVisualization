import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

alt.data_transformers.enable("vegafusion")

@st.cache_data
def load_data():
    # data = pd.read_csv('../Assignements/WDI_CSV_2025_01_28/WDICSV.csv')
    # countries = pd.read_csv('../Assignements/WDI_CSV_2025_01_28/WDICountry.csv')
    # df = pd.melt(data, id_vars=['Country Name', 'Country Code', 'Indicator Name',
    #              'Indicator Code'], var_name='Year', value_name='Value')
    # df = df.merge(countries[['Country Code', 'Region',
    #               'Income Group']], on='Country Code', how='left')
    # df['Year'] = df['Year'].astype(int)
    df = pd.read_csv('data/poverty_data.csv')
    return df

if "df" not in st.session_state:
    st.session_state["df"] = load_data()
df_copy = st.session_state["df"].copy()

### First Graph
# indicators = [
#     'Poverty gap at $2.15 a day (2017 PPP) (%)',
#     'Poverty gap at $3.65 a day (2017 PPP) (%)',
#     'Poverty gap at $6.85 a day (2017 PPP) (%)',
#     'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)',
#     'Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)',
#     'Poverty headcount ratio at $6.85 a day (2017 PPP) (% of population)',
#     'Poverty headcount ratio at national poverty lines (% of population)',
#     'Poverty headcount ratio at societal poverty line (% of population)',
#     'Multidimensional poverty headcount ratio (World Bank) (% of population)',
#     'Population, total',
#     'GDP per capita (constant 2015 US$)',
#     'GNI per capita (constant 2015 US$)'
# ]

regions = [
    'South Asia', 'Europe & Central Asia', 'Middle East & North Africa',
    'East Asia & Pacific', 'Sub-Saharan Africa', 'Latin America & Caribbean'
]

# Filter and pivot the data
df = df_copy[df_copy['Country Name'].isin(regions)].drop(columns='Region')
df = df.rename(columns={'Country Name': 'Region'})
df = pd.pivot_table(df, index=['Region', 'Year'], columns='Indicator Name', values='Value').reset_index()
df['Total Population (in million)'] = df['Population, total'] / 1_000_000
# Calculate the number of poor people
df['Number of Poor (in million)'] = (df['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)'] / 100) * df['Total Population (in million)']
# Filter and pivot the world data
df_world = df_copy[(df_copy['Country Name'] == 'World')].drop(columns='Region')
df_world = df_world.rename(columns={'Country Name': 'Region'})
df_world = pd.pivot_table(df_world, index=['Region', 'Year'], columns='Indicator Name', values='Value').reset_index()
df_world['Year'] = df_world['Year'].astype(int)
# Calculate the number of poor people for the world
df_world['Number of Poor (in million)'] = (df_world['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)'] / 100) * (df_world['Population, total'] / 1_000_000)
df_world['Total Population (in million)'] = df_world['Population, total'] / 1_000_000

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


# Chart 2
test = df.sort_values(by=['Region', 'Year'])
test = test.groupby('Region').ffill()

test = test.merge(df[['Region']], left_index=True, right_index=True)
poor = alt.Chart(test[(test.Year >= 1990) & (test.Year <= 2022)]).mark_area(opacity=0.7).encode(
    x='Year:O',
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of People')),
    color=alt.Color('Region:N')
).properties(
    title='Number of People Living in Extreme Poverty',
    width=600,
    height=300
)


# Chart 3
# Filter and pivot the data
df_country = df_copy[df_copy['Region'].notna()]
# df = df.rename(columns={'Country Name': 'Region'})
df_country = pd.pivot_table(df_country, index=['Country Name','Region', 'Year'], columns='Indicator Name', values='Value').reset_index()

# Convert 'Year' to integer and population to millions
df_country['Year'] = df_country['Year'].astype(int)
df_country['Total Population (in million)'] = df_country['Population, total'] / 1_000_000

# Calculate the number of poor people
df_country['Number of Poor (in million)'] = (df_country['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)'] / 100) * df_country['Total Population (in million)']
df_country['poverty ratio'] = df_country['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)']
test = df_country.sort_values(by=['Country Name', 'Year'])
test = test.groupby('Country Name').ffill()

test = test.merge(df_country[['Country Name']], left_index=True, right_index=True)

# Create a selection for the year
select_year = alt.selection_point(name='select', fields=['Year'], value=1991, bind=alt.binding_range(min=1991, max=2022, step=1))

# Create the Altair chart with adjusted size scale
chart3 = alt.Chart(test.dropna(subset='Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)')).mark_point(filled=True, ).encode(
    x=alt.X('poverty ratio:Q', scale=alt.Scale(type='linear',domain=(0.1, 100))),
    color=alt.Color('Region', scale=alt.Scale(scheme='tableau10')),
    size=alt.Size('Number of Poor (in million):Q', scale=alt.Scale(domain=[0,600], range=[100, 2000])),  # Adjust the size scale
    row= alt.Row('Region',header=alt.Header(labelAngle=0)),
    tooltip=['Country Name', 'poverty ratio:Q', 'Number of Poor (in million):Q', 'Region']
).properties(
    title='Number of Poor People in Different Countries',
    width=600,
    height=50
).interactive(bind_x=True, bind_y=True).add_params(
    select_year
).transform_filter(
    select_year
)



# Create a selection for the year
select_year = alt.selection_point(name='select_year', fields=['Year'], value=1991, bind=alt.binding_range(min=1991, max=2022, step=1))

# Create a selection for the country
select_country = alt.selection_point(name='select_country', fields=['Country Name'],value='China', on='click', empty='all')

# Create the first chart with adjusted size scale
chart = alt.Chart(test.dropna(subset='Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)')).mark_point(filled=True).encode(
    x=alt.X('poverty ratio:Q', scale=alt.Scale(type='linear', domain=(0.1, 100))),
    color=alt.Color('Region', scale=alt.Scale(scheme='tableau10')),
    size=alt.Size('Number of Poor (in million):Q', scale=alt.Scale(domain=[0, 400], range=[100, 2000])),  # Adjust the size scale
    row=alt.Row('Region', header=alt.Header(labelAngle=0)),
    tooltip=['Country Name', 'poverty ratio:Q', 'Number of Poor (in million):Q', 'Region']
).properties(
    title='Number of Poor People in Different Countries',
    width=400,
    height=50
).interactive(bind_x=True, bind_y=True).add_params(
    select_year, select_country
).transform_filter(
    select_year
).add_params(
    select_country
)

# Create the second chart to show the selected country's data
country = alt.Chart(test.dropna(subset='Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)')).mark_line().encode(
    x='Year:O',
    y='poverty ratio:Q',
    tooltip=['Country Name', 'poverty ratio', 'Number of Poor (in million)', 'Region']
).properties(
    title='Poverty Ratio Over Time for Selected Country',
    width=400,
    height=200
).transform_filter(
    select_country
)

# # Combine the charts vertically
# combined_chart = alt.hconcat(chart, country).resolve_scale(
#     color='independent'
# )

# combined_chart

# Add a text mark to simulate a dynamic title
title_text = alt.Chart(test).mark_text(
    align='center',
    fontSize=16,
    dy=-10
).encode(
    text=alt.Text('Country Name:N', title='Country')
).transform_filter(
    select_country
).properties(
    width=400,
    height=50
)

# Combine the charts horizontally with the title text
combined_chart = alt.vconcat(
    alt.hconcat(chart, alt.vconcat(title_text, country)),
).resolve_scale(
    color='independent'
)


st.title("Toward ending extreme poverty")
col1, col2 = st.columns([2, 1])
st.write(
    """SDG target 1.1 aims to end extreme poverty everywhere by 2030.
        The World Bank redefines extreme poverty as living on less than $2.15 a day. 
        Over the past decades, the world has made significant progress in ending extreme poverty while at the same time the world population has increased.""")
st.altair_chart(chart1) #, use_container_width=True)
st.write(""""
However this large reduction of global poverty, masks uneven progress.
 In fact most of the extreme poor toward the end of the last century lived in"
 East and South Asia where stark economic growth leads two significant reductions in poverty.
 On the contrary the number of extreme poor people in Sub-Saharan Africa has increased.""")
st.altair_chart(poor, use_container_width=True)

st.altair_chart(combined_chart)

