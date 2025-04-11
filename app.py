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
    poverty = pd.read_parquet('data/oda_poverty_world.parquet')
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
    height=500
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
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of extreme poor')),
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

# -- Chart 5: ODA Top Donors and Recipients --


# -- Chart 6: top donors and recipients over year
color_scale = alt.Scale(
    domain=['Low income', 'Lower middle income', 'Upper middle income', 'High income'],
    range=['#d62728', '#f7b6d2', '#76c7f0', '#1f77b4']
)

# Prepare the data
year_selection = alt.selection_point(
    name='Select',
    fields=['Year'],
    bind=alt.binding_range(min=1990, max=2022, step=1, name='Year: '),
    value=[{'Year': 2022}]
)

# Top Donors
top_donors = poverty[poverty['Region'].notna()].groupby(['Country Name', 'Year', 'Income Group'], as_index=False)["Net ODA provided, total (constant 2021 US$)"].sum()

top_donor_chart = alt.Chart(top_donors).transform_filter(
    year_selection
).transform_window(
    rank='rank(datum["Net ODA provided, total (constant 2021 US$)"])',
    sort=[alt.SortField("Net ODA provided, total (constant 2021 US$)", order='descending')]
).transform_filter(
    (alt.datum.rank <= 10)
).mark_bar().encode(
    x=alt.X('Net ODA provided, total (constant 2021 US$):Q', title='ODA Provided (2021 US$)', axis=alt.Axis(format=".2~s", labelExpr="replace(datum.label, 'G', 'B')")
),
    y=alt.Y('Country Name:N', sort='-x', title=None),
    tooltip=['Country Name:N', 'Net ODA provided, total (constant 2021 US$):Q'],
    color=alt.Color('Income Group:N', scale=color_scale, legend=alt.Legend(title="Income Group"))
).properties(
    title='Top 10 Donor Countries',
    width=300,
    height=300
)

recipient_selection = alt.selection_point(fields=['Country Name'], value=[{"Country Name":'Iraq'}])

# Top Recipients
top_recipients = poverty[poverty['Region'].notna()].groupby(['Country Name', 'Year', 'Income Group'], as_index=False)["Net official development assistance received (constant 2021 US$)"].sum()

top_recipient_chart = alt.Chart(top_recipients).transform_filter(
    year_selection
).transform_window(
    rank='rank(datum["Net official development assistance received (constant 2021 US$)"])',
    sort=[alt.SortField("Net official development assistance received (constant 2021 US$)", order='descending')]
).transform_filter(
    (alt.datum.rank <= 10)
).mark_bar().encode(
    x=alt.X('Net official development assistance received (constant 2021 US$):Q', title='ODA Received (2021 US$)', axis=alt.Axis(format=".2~s", labelExpr="replace(datum.label, 'G', 'B')")
),
    y=alt.Y('Country Name:N', sort='-x', title=None),
    tooltip=['Country Name:N', 'Net official development assistance received (constant 2021 US$):Q'],
    color=alt.Color('Income Group:N',  scale=color_scale, legend=alt.Legend(title="Income Group"))
).properties(
    title='Top 10 Recipient Countries',
    width=300,
    height=300
)

# Combine the charts and add the slider
top_donor_recipient_chart = (top_donor_chart | top_recipient_chart).add_params(
    year_selection
).resolve_scale(
    color='shared'
).configure_title(
    fontSize=16,
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).properties(title='Top 10 Donors and Recipients of Official Development Assistance (ODA) over years')

poverty["Net ODA provided, total (% of GNI)"] = poverty["Net ODA provided, total (% of GNI)"].fillna(poverty["Net ODA provided, total (current US$)"] / poverty["GNI (current US$)"] * 100)

top_donors_bis = poverty[poverty['Region'].notna()].groupby(['Country Name', 'Year', 'Income Group'], as_index=False)["Net ODA provided, total (% of GNI)"].sum()

top_donor_chart_bis = alt.Chart(top_donors_bis).transform_filter(
    year_selection
).transform_window(
    rank='rank(datum["Net ODA provided, total (% of GNI)"])',
    sort=[alt.SortField("Net ODA provided, total (% of GNI)", order='descending')]
).transform_filter(
    (alt.datum.rank <= 10)
).mark_bar().encode(
    x=alt.X('Net ODA provided, total (% of GNI):Q', title='ODA Provided (% of GNI)'),
    y=alt.Y('Country Name:N', sort='-x', title=None),
    tooltip=['Country Name:N', 'Net ODA provided, total (% of GNI):Q'],
    color=alt.Color('Income Group:N', scale=color_scale, legend=alt.Legend(title="Income Group"))
).properties(
    title='Top 10 Donor Countries (for ODA as % of GNI)',
    width=300,
    height=300
)
# Create vertical rule chart at x = 0.7
target_line = alt.Chart(pd.DataFrame({'x': [0.7]})).mark_rule(color='red', strokeWidth=2).encode(
    x=alt.X('x:Q')
)

# Create text label chart positioned near the vertical line
target_text = alt.Chart(pd.DataFrame({'x': [0.7], 'label': ['target 0.7% GNI']})).mark_text(
    fontSize=14,
    align='left',
    dx=5,  # horizontal offset
    dy=-5,  # vertical offset; adjust as needed
    color='red'
).encode(
    x=alt.X('x:Q'),
    text='label:N'
)

# Combine the original chart with the target vertical rule and the label
final_donor_chart_bis= top_donor_chart_bis + target_line + target_text
# recipient_selection = alt.selection_point(fields=['Country Name'], value=[{"Country Name":'Iraq'}])

# Top Recipients
top_recipients_bis = poverty[poverty['Region'].notna()].groupby(['Country Name', 'Year', 'Income Group'], as_index=False)['Net ODA received per capita (current US$)'].sum()

top_recipient_chart_bis = alt.Chart(top_recipients_bis).transform_filter(
    year_selection
).transform_window(
    rank='rank(datum["Net ODA received per capita (current US$)"])',
    sort=[alt.SortField("Net ODA received per capita (current US$)", order='descending')]
).transform_filter(
    (alt.datum.rank <= 10)
).mark_bar().encode(
    x=alt.X('Net ODA received per capita (current US$)', title='ODA Received per capita', axis=alt.Axis(format=".2~s", labelExpr="replace(datum.label, 'G', 'B')")
),
    y=alt.Y('Country Name:N', sort='-x', title=None),
    tooltip=['Country Name:N', 'Net ODA received per capita (current US$)'],
    color=alt.Color('Income Group:N',  scale=color_scale, legend=alt.Legend(title="Income Group"))
).properties(
    title='Top 10 Recipient Countries',
    width=300,
    height=300
)

# Combine the charts and add the slider
top_donor_recipient_chart_bis = (final_donor_chart_bis | top_recipient_chart_bis).add_params(
    year_selection
).resolve_scale(
    color='shared'
).configure_title(
    fontSize=16,
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).properties(title='Top 10 Donors and Recipients of Official Development Assistance (ODA) over years')


# --- Chart 7: ODA Financial Flows Between Regions ---

regions = ['Europe & Central Asia', 'Middle East & North Africa', 'Sub-Saharan Africa',
 'Latin America & Caribbean', 'South Asia','North America', 'East Asia & Pacific']

# Sample code for processing the data (you already have this part in your script)
df_regions = oda[
    (oda['Recipient_Region'].isin(regions)) &
    (oda['Donor_Region'].isin(regions)) &
    (oda['Aid type'] == 'Memo: ODA Total, Gross disbursements')
]

df_regions['Value_k'] = df_regions['Value'] / 1000  # Dividing by 1000 for easier plotting
df_regions_1990 = df_regions[df_regions['Year'] == 1990]
df_regions_2021 = df_regions[df_regions['Year'] == 2021]

grouped_1990 = df_regions_1990.groupby(['Donor_Region', 'Recipient_Region'], as_index=False)['Value_k'].sum()
flow_matrix_1990 = grouped_1990.pivot(index='Donor_Region', columns='Recipient_Region', values='Value_k').fillna(0)
grouped_2021 = df_regions_2021.groupby(['Donor_Region', 'Recipient_Region'], as_index=False)['Value_k'].sum()
flow_matrix_2021 = grouped_2021.pivot(index='Donor_Region', columns='Recipient_Region', values='Value_k').fillna(0)
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
circos_1990 = Circos.chord_diagram(
    flow_matrix_1990,
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

circos_2021 = Circos.chord_diagram(
    flow_matrix_2021,
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
circos_1990.text(f"ODA Financial Flows Between Regions in 1990", deg=0, r=150, size=14)
fig_1990 = circos_1990.plotfig()
circos_2021.text(f"ODA Financial Flows Between Regions in 2021", deg=0, r=150, size=14)
fig_2021 = circos_2021.plotfig()


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
    st.altair_chart(chart1)
    st.markdown("**Source:** World Bank,World Development indicator Database (WDI)")
with col2:
    st.altair_chart(chart2)

st.markdown("**Note:** The number of extreme poor is calculated using the poverty headcount ratio at $2.15 a day (2017 PPP) and the total population.")


# --- Chart x: ODA fLows


#############################################
# --- Part 2: GDP and Poverty Reduction --- #
#############################################





#########################################################
# --- Part 3: Financial Flows and Poverty Reduction --- #
#########################################################

st.markdown(""" Economic growth alone doesn't explain everything. Let's dig deeper into the role of international financial flows,
such as aid, investments, and remittances, in shaping poverty outcomes worldwide.""")
st.header("Part 2: Global Partnership and cooperations to end poverty")

st.subheader("Section 1: Global Financial Flows — The Big Picture")
st.markdown(
"""
To understand how countries combat poverty, it is interesting to look at the financiallifelines flowing across borders like:           
- **Net Official Development Assistance (ODA):** Grants or concessional loans from government to support development.
- **Foreign Direct Investment (FDI):** Cross-border investments, indicating private sector engagement.
- **Personal Remittances:** Funds sent by individuals (often migrant workers) to their home countries.
.""")

st.altair_chart(world_financial_aid)
st.markdown(
'''
### Key Observations:
- **Foreign Direct Investment (FDI)** has become the **dominant source of external finance** over the years. Unlike aid, FDI often targets infrastructure, industries, and services that stimulate long-term economic growth and employment creation.
- **Remittances** have experienced a **remarkable surge**, outpacing the growth of traditional aid flows.  This growth reflects both the increase in migration and the resilience of remittance flows during crises.
- **Official Development Assistance (ODA)** remains vital, especially for countries with limited access to private finance, but its relative share has decreased compared to private flows.


### Aid Types Evolution by Region (1990–2022)
''')
st.altair_chart(world_aid_type)

st.markdown('''

A notable trend emerges when observing the evolution of aid types: **Imputed Multilateral ODA** has steadily increased since the late 1990s, surpassing other aid types to become the largest component of aid flows by the early 2000s. 

This growth reflects the **increasing importance of multilateral institutions** in the global development landscape. Countries have progressively channeled more resources through organizations like the **United Nations**, **World Bank**, and **regional development banks**, recognizing the need for coordinated, impartial, and large-scale responses to complex global challenges.

#### Regional Disparities & Notable Events:
- **Caribbean & Central America (2010):**  
  A dramatic spike in **Humanitarian Aid** corresponds with the devastating earthquake in **Haiti** in January 2010.  
  The international community responded with unprecedented humanitarian assistance to support emergency relief and reconstruction efforts.    
- **Middle East:** Periodic increases in **Humanitarian Aid** can be linked to conflicts and refugee crises in the region, such as the Syrian conflict starting in 2011.
- **Europe**: Surge in aid flows in 2022 to support Ukraine in the conflict.

*This chart emphasizes how aid flows respond dynamically to regional crises, geopolitical events, and development priorities.*
'''
)

st.markdown("In this section, we dive deeper into the distribution of Official Development Assistance (ODA).")
tab1, tab2 = st.tabs(["Absolute ODA", "Relative ODA"])
tab1.altair_chart(top_donor_recipient_chart, use_container_width=True)
tab2.altair_chart(top_donor_recipient_chart_bis, use_container_width=True)

st.markdown("Even though official aid has grown over the years, it still falls significantly short of the 0.7% of GNI benchmark expected from developed nations to support less wealthy countries, as outlined in SDG target 17.2. Only 4 countries achieved this goal in 2022 ")

st.subheader("Section 2: Bilateral ODA Flows Between Regions")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_1990)
with col2:
    st.pyplot(fig_2021)


year_selection2 = alt.selection_point(
    name='Select year',
    fields=['Year'],
    bind=alt.binding_range(min=1990, max=2022, step=1, name='Year: '),
    value=[{'Year': 2022}]
)

brush = alt.selection_interval()

poverty['poverty headcount ratio']= poverty['Poverty headcount ratio at $2.15 a day']
x_domain = [1, 1000]  # Adjust these values as needed
y_domain = [0, 100]   # Adjust these values as needed

scatter_poverty_oda = alt.Chart(poverty[poverty['Region'].notna()]).mark_point().encode(
    x=alt.X('Net ODA received per capita (current US$):Q',
            scale=alt.Scale(domain=x_domain, type='log'),
            axis=alt.Axis(title='Net ODA per Capita (US$)')),
    y=alt.Y('poverty headcount ratio:Q',
            scale=alt.Scale(domain=y_domain),
            axis=alt.Axis(title='Poverty Headcount Ratio (%)')),
    color=alt.Color('Income Group:N',
                    scale=color_scale,
                    legend=alt.Legend(title="Income Group")),
    tooltip=['Country Name:N', 'Year:O',
             'Net ODA received per capita (current US$):Q',
             'poverty headcount ratio:Q']
).add_params(
    year_selection2,
    brush
).transform_filter(
    year_selection2  # Apply the year selection filter
).properties(
    height=400,
    width=400,
    title="Poverty Headcount Ratio at $2.15/day vs ODA per Capita"
)

bar_oda_plot = alt.Chart(poverty[poverty['Region'].notna()]).mark_bar().encode(
    y='Country Name:N',
    x="Net official development assistance received (constant 2021 US$):Q",
    color=alt.Color('Income Group:N',
                    scale=color_scale,
                    legend=alt.Legend(title="Income Group")),
    tooltip=['Country Name:N', 'Year:O','Net official development assistance received (constant 2021 US$):Q']
).add_params(year_selection2).transform_filter(brush, year_selection2).properties(
    height=500,
    width=300,
)

col1, col2 = st.columns(2)
st.altair_chart(scatter_poverty_oda | bar_oda_plot)

