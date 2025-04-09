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
    # df_2021 = pd.read_csv('./data/oda_2021.csv')
    oda = pd.read_parquet('./data/oda_disbursment.parquet')
    # df_2021 = data[data.Year == 2021].dropna(subset=['Value'])
    print('ODA data loaded')
    countries = pd.read_csv('./data/WDICountry.csv').rename(columns={"Country Code":"DE_code"})
    print("countries data loaded")
    correspondance = pd.read_csv('./data/Correspondences_DAC2a.csv')
    print('correspondance data loaded')
    poverty = pd.read_csv('data/oda_poverty_world.csv')
    return oda, countries, correspondance, poverty

oda, countries, correspondance, poverty = load_data()

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



#########################################################
# --- Part 3: Financial Flows and Poverty Reduction --- #
#########################################################

# -- - Chart 3: Financial Flows to the World ---
world_financial_aid_flows = poverty[(poverty['Country Name']=='World') & (poverty['Year'] >= 1990) & (poverty['Year'] <= 2022)]
world_financial_aid_flows = world_financial_aid_flows[['Country Name','Year',"Net official development assistance received (current US$)",
'Foreign direct investment, net inflows (BoP, current US$)','Personal remittances, received (current US$)']]
world_financial_aid_flows = pd.melt(world_financial_aid_flows, id_vars=['Country Name','Year'], var_name='Indicator', value_name='Value')
world_financial_aid_flows.loc[:,'Value_M'] = world_financial_aid_flows.loc[:,'Value'] / 1000_000_000
world_financial_aid = alt.Chart(world_financial_aid_flows).mark_line().encode(
    x='Year:O',
    y=alt.Y('Value_M:Q',title='Flow in Billion US$'),
    color=alt.Color('Indicator:N', legend=alt.Legend(
        columns=1,
        orient='bottom',  
        title='Financial Aid Indicators',
        labelLimit=500,  
        direction='vertical',
        symbolSize=150
    )),
    tooltip=['Year:O', 'Indicator:N', 'Value_M:Q']
).properties(
    width=500,
    height=500,
    title='Global Financial flows Over Time'
).interactive()


# -- Chart 4: Different Types of Aid Over Time and Recipients --
aid_types = ['Imputed Multilateral ODA', 'Technical Cooperation', 'Development Food Aid', 'Humanitarian Aid']

# recipients =  
recipients = ['All Recipients, Total','Western Africa, Total','South of Sahara, Total',
 'Southern Africa, Total', 'South & Central Asia, Total','South America, Total','Oceania, Total',
'North of Sahara, Total', 'Northern America, Total','Middle East, Total','Fragile states, Total',
'Europe, Total','Eastern Africa, Total','Developing Countries, Total','Caribbean & Central America, Total',
'Asia, Total','All Recipients, Total','Africa, Total','LDCs, Total',
 'LMICs, Total']

recipient_dropdown = alt.selection_point(
    fields=['Recipient'],
    bind=alt.binding_select(options=sorted(recipients), name='Select a Recipient: '),
    value=[{'Recipient':'All Recipients, Total'}]
)

# recipients_dropdown
oda_world = oda[
    (oda['Recipient'].isin(recipients)) & 
    (oda['Year'] >= 1990) & 
    (oda['Year'] <= 2022) & 
    (oda['Aid type'].isin(aid_types))
]

# Aggregate and convert to billion $
oda_agg = oda_world.groupby(['Year', 'Aid type','Recipient'], as_index=False)['Value'].sum()
oda_agg['Value_k'] = oda_agg['Value'] / 1000.0

world_aid_type = alt.Chart(oda_agg).mark_line().encode(
    x='Year:O',
    y=alt.Y('Value_k:Q',title='Flow in million US$'),
    color=alt.Color('Aid type:N'),
    tooltip=['Year:O', 'Aid type:N', 'Value_k:Q']
).interactive().properties(
    width=200,
    height=400,
    title='Evolution of Different Types of Aid Over Time by Region'
).add_params(
    recipient_dropdown
).transform_filter(
    recipient_dropdown
)

#################################################
## ------------ Streamlit Display ------------ ##
#################################################

st.title("🌍 How economic growth and financial aid shape the fight against poverty?")

##############################################
# --- Part 1: Overview of Global Poverty --- #
##############################################
st.header("Part 1: Global Overview: Population Growth & Poverty Reduction")

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

#############################################
# --- Part 2: GDP and Poverty Reduction --- #
#############################################





#########################################################
# --- Part 3: Financial Flows and Poverty Reduction --- #
#########################################################

st.markdown(""" Economic growth alone doesn't explain everything. Let's dig deeper into the role of international financial flows,
such as aid, investments, and remittances, in shaping poverty outcomes worldwide.""")
st.header("Part 2: Understanding financial flows and their impact on poverty reduction")

st.subheader("Section 1: Global Financial Flows — The Big Picture")
st.markdown("""To understand how countries combat poverty, it is interesting to look at the financial
             lifelines flowing across borders like:
            - **Net Official Development Assistance (ODA):** Grants or concessional loans from government to support development.
            - **Foreign Direct Investment (FDI):** Cross-border investments, indicating private sector engagement.
            - **Personal Remittances:** Funds sent by individuals (often migrant workers) to their home countries.
.""")

# How have different financial flows evolved over time?
# Are we seeing a growing reliance on private flows (remittances, FDI) versus official aid?
# Does global aid increase in times of crisis?

st.altair_chart(world_financial_aid, use_container_width=True)
st.markdown(
'''
### Key Observations:
- **Foreign Direct Investment (FDI)** has become the **dominant source of external finance** over the years. Unlike aid, FDI often targets infrastructure, industries, and services that stimulate long-term economic growth and employment creation.
- **Remittances** have experienced a **remarkable surge**, outpacing the growth of traditional aid flows.  
  This growth reflects both the increase in migration and the resilience of remittance flows during crises.  
  For example, during economic downturns or disasters, remittances tend to remain stable or even increase, as migrants support their families back home.
- **Official Development Assistance (ODA)** remains vital, especially for countries with limited access to private finance, but its relative share has decreased compared to private flows.

*Overall, this shift highlights how global development financing has diversified beyond aid, with private sector flows and diaspora remittances playing increasingly important roles.*

---

# Aid Types Evolution by Region (1990–2022)
''')


st.altair_chart(world_aid_type, use_container_width=True)

st.markdown('''

A notable trend emerges when observing the evolution of aid types: **Imputed Multilateral ODA** has steadily increased since the late 1990s, surpassing other aid types to become the largest component of aid flows by the early 2000s. 

This growth reflects the **increasing importance of multilateral institutions** in the global development landscape. Countries have progressively channeled more resources through organizations like the **United Nations**, **World Bank**, and **regional development banks**, recognizing the need for coordinated, impartial, and large-scale responses to complex global challenges.

### Regional Disparities & Notable Events:

- **Caribbean & Central America (2010):**  
  A dramatic spike in **Humanitarian Aid** corresponds with the devastating earthquake in **Haiti** in January 2010.  
  The international community responded with unprecedented humanitarian assistance to support emergency relief and reconstruction efforts.
            
- **Middle East:**  
  Periodic increases in **Humanitarian Aid** can be linked to conflicts and refugee crises in the region, such as the Syrian conflict starting in 2011.

*This chart emphasizes how aid flows respond dynamically to regional crises, geopolitical events, and development priorities.*

End **reflection**
   - "How might future aid and private flows evolve as global challenges like climate change and pandemics shape development priorities?"
'''
)