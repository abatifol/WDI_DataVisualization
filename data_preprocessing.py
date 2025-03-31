import pandas as pd
import streamlit as st


# We load and extract relevant data from the dataset beforehand to create a smaller dataset
# to make the streamlit app faster and more efficient.

data = pd.read_csv('../Assignements/WDI_CSV_2025_01_28/WDICSV.csv')
countries = pd.read_csv('../Assignements/WDI_CSV_2025_01_28/WDICountry.csv')
df = pd.melt(data, id_vars=['Country Name', 'Country Code', 'Indicator Name',
                'Indicator Code'], var_name='Year', value_name='Value')
df = df.merge(countries[['Country Code', 'Region',
                'Income Group']], on='Country Code', how='left')
df['Year'] = df['Year'].astype(int)

indicators = [
    'Poverty gap at $2.15 a day (2017 PPP) (%)',
    'Poverty gap at $3.65 a day (2017 PPP) (%)',
    'Poverty gap at $6.85 a day (2017 PPP) (%)',
    'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)',
    'Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)',
    'Poverty headcount ratio at $6.85 a day (2017 PPP) (% of population)',
    'Poverty headcount ratio at national poverty lines (% of population)',
    'Poverty headcount ratio at societal poverty line (% of population)',
    'Multidimensional poverty headcount ratio (World Bank) (% of population)',
    'Population, total',
    'GDP per capita (constant 2015 US$)',
    'GNI per capita (constant 2015 US$)'
]

df = df[df['Indicator Name'].isin(indicators)]

df.to_csv('data/poverty_data.csv', index=False)
