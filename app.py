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
    # oda = pd.read_parquet('./data/oda_disbursment.parquet')
    donor_prepared = pd.read_parquet('data/chart8_donor_prepared.parquet')
    recipient_prepared = pd.read_parquet('data/chart8_recipient_prepared.parquet')
    chart4_oda_agg = pd.read_csv('data/chart4_oda_agg.csv')
    chart7_oda_df_regions = pd.read_csv('data/chart7_oda_df_regions.csv')
    # df_2021 = data[data.Year == 2021].dropna(subset=['Value'])
    print('ODA data loaded')
    countries = pd.read_parquet('./data/WDICountry.parquet').rename(columns={"Country Code":"DE_code"})
    print("countries data loaded")
    correspondance = pd.read_csv('./data/Correspondences_DAC2a.csv')
    print('correspondance data loaded')
    poverty = pd.read_parquet('data/oda_poverty_world.parquet')
    df = pd.read_csv("./data/GDP_Poverty_Gini_Unemployment_Enriched.csv")
    return countries, correspondance, poverty, chart4_oda_agg, chart7_oda_df_regions, donor_prepared, recipient_prepared, df

countries, correspondance, poverty, chart4_oda_agg, chart7_oda_df_regions, donor_prepared, recipient_prepared, df = load_data()

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

# df_world.to_csv('data/chart1_df_world.csv', index=False)
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
        text='World Population and extreme poverty', 
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


regions_color = alt.Scale(
    domain = regions,
    range = ['#ffd60a','#0000ff','#ff7f00','#7400b8','#f94144','#00ff00']
)

df_pov_regions = poverty[poverty['Country Name'].isin(regions)].drop(columns='Region')
df_pov_regions = df_pov_regions.rename(columns={'Country Name': 'Region'})
temp = df_pov_regions.sort_values(by=['Region', 'Year'])
# fill the missing years with the last known value
temp = temp.groupby('Region').ffill()
df_pov_regions = temp.merge(df_pov_regions[['Region']], left_index=True, right_index=True)

# df_pov_regions.to_csv('data/chart2_df_pov_regions.csv', index=False)

chart_regions_pov = alt.Chart(df_pov_regions[(df_pov_regions.Year >= 1990) & (df_pov_regions.Year <= 2022)]).mark_area(opacity=0.7).encode(
    x='Year:O',
    y=alt.Y('Number of Poor (in million):Q', axis=alt.Axis(title='Millions of extreme poor')),
    color=alt.Color(
        'Region:N',
        scale=regions_color, #alt.Scale(scheme='turbo'),
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
# Create vertical rule chart at x = 0.7
target_line_covid = alt.Chart(pd.DataFrame({'x': [2020]})).mark_rule(color='black', strokeWidth=1).encode(
    x=alt.X('x:Q')
)

# Create text label chart positioned near the vertical line
target_text_covid = alt.Chart(pd.DataFrame({'label': ['COVID19']})).mark_text(
    fontSize=14,
    align='right',
    dx=400,  # horizontal offset
    dy=140,  # vertical offset; adjust as needed
    color='black'
).encode(
    text='label:N'
)
chart2 = (chart_regions_pov + target_line_covid + target_text_covid)

#########################################################
# --- Part 3: Financial Flows and Poverty Reduction --- #
#########################################################

# -- - Chart 3: Financial Flows to the World ---
world_financial_aid_flows = poverty[(poverty['Country Name']=='World') & (poverty['Year'] >= 1990) & (poverty['Year'] <= 2022)]
world_financial_aid_flows = world_financial_aid_flows[['Country Name','Year',"Net official development assistance received (current US$)",
'Foreign direct investment, net inflows (BoP, current US$)','Personal remittances, received (current US$)']]
world_financial_aid_flows = pd.melt(world_financial_aid_flows, id_vars=['Country Name','Year'], var_name='Indicator', value_name='Value')
world_financial_aid_flows.loc[:,'Value_M'] = world_financial_aid_flows.loc[:,'Value'] / 1000_000_000

# world_financial_aid_flows.to_csv('data/chart3_world_financial_aid_flows.csv', index=False)

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
'Asia, Total','Africa, Total','LDCs, Total',
 'LMICs, Total']

recipient_dropdown = alt.selection_point(
    fields=['Recipient'],
    bind=alt.binding_select(options=sorted(recipients), name='Select a Recipient: '),
    value=[{'Recipient':'All Recipients, Total'}]
)

# recipients_dropdown
# oda_world = oda[
#     (oda['Recipient'].isin(recipients)) & 
#     (oda['Year'] >= 1990) & 
#     (oda['Year'] <= 2022) & 
#     (oda['Aid type'].isin(aid_types))
# ]

# # Aggregate and convert to billion $
# oda_agg = oda_world.groupby(['Year', 'Aid type','Recipient'], as_index=False)['Value'].sum()
# oda_agg['Value_k'] = oda_agg['Value'] / 1000.0

# oda_agg.to_csv('data/chart4_oda_agg.csv', index=False)
# chart4_oda_agg = oda_agg.copy()
world_aid_type = alt.Chart(chart4_oda_agg).mark_line().encode(
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
# df_regions = oda[
#     (oda['Recipient_Region'].isin(regions)) &
#     (oda['Donor_Region'].isin(regions)) &
#     (oda['Aid type'] == 'Memo: ODA Total, Gross disbursements')
# ]

# df_regions.to_csv('data/chart7_oda_df_regions.csv', index=False)
# chart7_oda_df_regions = df_regions.copy()
chart7_oda_df_regions['Value_k'] = chart7_oda_df_regions['Value'] / 1000  # Dividing by 1000 for easier plotting


df_regions_1990 = chart7_oda_df_regions[chart7_oda_df_regions['Year'] == 1990]
df_regions_2021 = chart7_oda_df_regions[chart7_oda_df_regions['Year'] == 2021]

grouped_1990 = df_regions_1990.groupby(['Donor_Region', 'Recipient_Region'], as_index=False)['Value_k'].sum()
flow_matrix_1990 = grouped_1990.pivot(index='Donor_Region', columns='Recipient_Region', values='Value_k').fillna(0)
grouped_2021 = df_regions_2021.groupby(['Donor_Region', 'Recipient_Region'], as_index=False)['Value_k'].sum()
flow_matrix_2021 = grouped_2021.pivot(index='Donor_Region', columns='Recipient_Region', values='Value_k').fillna(0)


colors = {
    'Europe & Central Asia': '#0000ff',
    'Middle East & North Africa': '#f94144',
    'Sub-Saharan Africa': '#ffd60a',
    'Latin America & Caribbean': '#00ff00',
    'South Asia': '#ff7f00',
    'East Asia & Pacific': '#7400b8',
    'North America': '#76c7f0',
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

# --- Chart 8: Deep dive into each Recipient and Donor ODA distribution
# Define the selectors using arrays for the default values
donor_selector = alt.selection_point(
    name='SelectDonor',
    fields=['Donor'],
    value=[{'Donor': 'France'}],
    # clear='click'
)

recipient_selector = alt.selection_point(
    name='SelectRecipient',
    fields=['Recipient'],
    value=[{'Recipient': 'Türkiye'}],
    # clear='click'
)

# Donor Pie Chart: displays all donors contributing to the selected recipient.
pie_donor = alt.Chart(recipient_prepared).mark_arc().encode(
    theta=alt.Theta('sum(Value):Q', title="Contribution Amount"),
    color=alt.Color('Donor:N', title="Contributing Donors", scale=alt.Scale(scheme='category20')),
    tooltip=[
        alt.Tooltip('Donor:N', title='Donor'),
        alt.Tooltip('sum(Value):Q', title='Total Contribution', format='$,.0f')
    ]
).add_params(
    donor_selector
).transform_filter(
    # Filter the data to include only rows that match the selected recipient.
    recipient_selector
).properties(
    width=500,
    height=500
)

# Recipient Pie Chart: displays recipients aid flows for the selected donor.
pie_recipient = alt.Chart(donor_prepared).mark_arc().encode(
    theta=alt.Theta('sum(Value):Q', title="ODA disbursment"),
    color=alt.Color('Recipient:N', title="Main Recipients", scale=alt.Scale(scheme='category20')),
    tooltip=[
        alt.Tooltip('Recipient:N', title='Recipient'),
        alt.Tooltip('sum(Value):Q', title='ODA disbursment', format='$,.0f')
    ]
).add_params(
    recipient_selector
).transform_filter(
    # Filter the data to include only rows that match the selected donor.
    donor_selector
).properties(
    width=500,
    height=500
)

donor_title = alt.Chart(donor_prepared).transform_filter(
    donor_selector
).transform_aggregate(
    selDonor='max(Donor)',
    totalValue='sum(Value)',
    groupby=[]
).transform_calculate(
    title='datum.selDonor + "\'s main recipients in 2021 ($" + format(datum.totalValue, ",.0f")  + " Million)"'
).mark_text(
    align='center',
    fontSize=14,
    fontWeight='bold'
).encode(
    text='title:N'
).properties(width=400, height=30)


# Recipient Title: uses recipient_selector to display the selected recipient.
recipient_title = alt.Chart(recipient_prepared).transform_filter(
    recipient_selector
).transform_aggregate(
    selRecipient='max(Recipient)',
    totalValue='sum(Value)',
    groupby=[]
).transform_calculate(
    title='datum.selRecipient + "\'s main donors in 2021 ($" + format(datum.totalValue, ".2f") + " Million)"'
).mark_text(
    align='center',
    fontSize=14,
    fontWeight='bold'
).encode(
    text='title:N'
).properties(width=400, height=30)



bilateral_oda = (recipient_title & pie_donor) | (donor_title & pie_recipient)
bilateral_oda = bilateral_oda.resolve_scale(color='independent').properties(
    title='Distribution of bilateral ODA in 2021 country level in Million USD'
).configure_title(
    anchor='middle',  # 👈 this centers the title
    fontSize=18,
    fontWeight='bold'
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
    st.altair_chart(chart2)

st.markdown("**Note:** The number of extreme poor is calculated using the poverty headcount ratio at $2.15 a day (2017 PPP) and the total population.")


# --- Chart x: ODA fLows


#############################################
# --- Part 2: GDP and Poverty Reduction --- #


# =========================================================
# 🌍 Axis 1 – Final Streamlit Narrative (Improved v2)
# =========================================================


# Chargement du fichier enrichi

df = df[df["Year"].between(1995, 2022)]

st.subheader("📊 Section 2 – Economic Growth, Poverty & Inequality")

st.markdown("""
### Introduction

This section explores one of the two key forces behind these shifts: **economic growth**. Using GDP per capita, sectoral composition, and inequality, we unpack how economic development patterns shaped global poverty trajectories.
""")
st.markdown("""
### 🌍 Axis 1 — Economic Growth and Poverty Reduction
Over the past few decades, while global **GDP per capita** has been rising steadily, the world has witnessed an unprecedented **reduction in extreme poverty**.
However, this overall trend hides profound **regional disparities**.
In this section, we focus on the **role of economic growth** — both in terms of magnitude and structure — in shaping the evolution of poverty.
Through interactive visualizations, we investigate how GDP dynamics, sectoral transformations, and inequality levels interact with poverty outcomes.

Why did **some countries lift millions out of poverty**, while others saw little progress? Let’s explore.
""")
st.markdown("Explore how **GDP growth**, **economic structure**, and **inequality** impact **poverty reduction** across countries and over time.")

# =========================================================
# 1. GDP per Capita & Poverty Timeline – Multi-country
# =========================================================
st.header("1️⃣ GDP per Capita & Poverty Rate Over Time")
st.markdown("""
#### Economic Takeoff and Poverty Decline — Or Not?
This dual-line chart enables you to explore the **parallel trajectories of GDP per capita and poverty rate** across selected countries.
**China** showcases sustained economic expansion with massive poverty alleviation.
**India** shows strong improvements, while **Nigeria’s GDP per capita** grew modestly with poverty reduction lagging.
This highlights that **growth is essential**, but not always **inclusive**.
""")

countries = st.multiselect("Select one or more countries", sorted(df["Country Name"].unique()), default=["China", "India", "Nigeria"])
df_countries = df[df["Country Name"].isin(countries)]

col1, col2 = st.columns(2)

with col1:
    fig_gdp = px.line(
        df_countries,
        x="Year", y="GDP per capita",
        color="Country Name",
        markers=True,
        log_y=True
    )
    st.plotly_chart(fig_gdp, use_container_width=True)

with col2:
    fig_pov = px.line(
        df_countries,
        x="Year", y="Poverty rate",
        color="Country Name",
        markers=True
    )
    fig_pov.update_traces(mode="lines+markers", connectgaps=True)  # 🔧 Lignes + connexion entre points
    st.plotly_chart(fig_pov, use_container_width=True)

# =========================================================
# 2. GDP Sectoral Composition – Multi-country (stacked area)
# =========================================================
st.header("2️⃣ GDP Composition by Sector")
st.markdown("""
#### 🏭 Structural Transformation: A Key to Inclusion?
**Economic development** often entails a shift from **agriculture to industry and services**, which brings **productivity gains** and **poverty reduction**.
Use this chart to track how countries’ **economic engines are evolving — or stuck**.
""")

df_stack = df[df["Country Name"].isin(countries)][["Country Name", "Year", "Agriculture (% of GDP)", "Industry (% of GDP)", "Services (% of GDP)"]].dropna()
df_stack = df_stack.melt(id_vars=["Country Name", "Year"], var_name="Sector", value_name="Share of GDP")

fig_stack = px.area(
    df_stack,
    x="Year", y="Share of GDP", color="Sector",
    facet_col="Country Name", facet_col_wrap=2,
    title="Sectoral GDP Composition Over Time"
)
st.plotly_chart(fig_stack, use_container_width=True)

# =========================================================
# 3. Interactive Scatter – Sector vs Poverty Rate (Dynamic)
# =========================================================
# =========================================================
# 3. Interactive Scatter – Sector vs Poverty Rate (Dynamic)
# =========================================================
st.header("3️⃣ Sectoral Share vs Poverty Rate")
st.markdown("*🔵 Bubble size = Total population of each country in the selected year.*")
st.markdown("""
#### Sectoral Share and Poverty: A Hidden Relationship
This scatterplot explores the **relationship between the GDP share of a sector** (Agriculture, Industry, Services) and the **poverty rate**.
**High service sector shares** often correlate with **lower poverty**, while **agriculture-heavy economies** tend to be more vulnerable.
""")
year_select = st.slider("Choose Year", 1995, 2022, 2015)
sector_select = st.selectbox("Select sector share", [
    "Agriculture (% of GDP)",
    "Industry (% of GDP)",
    "Services (% of GDP)"
])
df_filtered = df[df["Year"] == year_select].dropna(subset=["Poverty rate", sector_select, "Population", "GDP per capita"])

fig_sector = px.scatter(
    df_filtered,
    x=sector_select, y="Poverty rate",
    size="Population", color="GDP per capita",
    color_continuous_scale="Plasma",
    range_color=[0, 40000], 
    size_max=65, 
    hover_name="Country Name",
    labels={"Poverty rate": "Poverty Rate (%)"},
    title=f"{sector_select} vs Poverty Rate ({year_select})"
)
st.plotly_chart(fig_sector, use_container_width=True)
# =========================================================
# 4. Animated Scatter – GDP per capita vs Poverty
# =========================================================
st.header("4️⃣ Animated Scatter – Growth & Poverty Over Time")
st.markdown("*🔵 Bubble size = Population of the country for each year (animation). Helps highlight the largest demographic impacts.*")
st.markdown("""
#### 🎥 A Moving Landscape — GDP, Poverty, and Inequality
This animated chart captures the **real-time evolution** of GDP per capita, poverty rate, and the Gini index across countries.
**China** and **India** show dramatic upward mobility. Other regions remain stagnant.
**Growth and equity** don’t always go hand in hand.
""")

df_anim = df.dropna(subset=["GDP per capita", "Poverty rate", "Population", "Gini Index"])
df_anim = df_anim[df_anim["GDP per capita"] > 0]
df_anim["Year"] = df_anim["Year"].astype(str)  # convert to string for animation
df_anim = df_anim.sort_values(by="Year")  # ensure correct order

fig_anim = px.scatter(
    df_anim,
    x="GDP per capita", y="Poverty rate",
    animation_frame="Year",
    size="Population", color="Gini Index",
    hover_name="Country Name", hover_data=["GDP per capita"],
    log_x=False,
    range_y=[0, 80],
    range_x=[100, 50000],  # Moins large
    size_max=80,           # Bulles plus grosses
    title="GDP per Capita vs Poverty Rate Over Time (colored by Gini Index)"
)

fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600 
fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
st.plotly_chart(fig_anim, use_container_width=True)

# =========================================================
# 5. Global Map – Gini or Poverty Rate
# =========================================================
st.header("5️⃣ Global Map – Inequality or Poverty")
st.markdown("""
#### 🗺️ Global Inequality & Poverty Snapshot
This choropleth map displays either **poverty rates** or **inequality levels (Gini Index)** for a selected year.
Notice how regions like **Latin America** or **Sub-Saharan Africa** remain hotspots of inequality.
""")

map_year = st.slider("Select year for the map", 1995, 2022, 2015)
map_metric = st.radio("Indicator to display:", ["Poverty rate", "Gini Index", "GDP per capita"])
df_map = df[df["Year"] == map_year].dropna(subset=[map_metric, "GDP per capita"])

fig_map = px.choropleth(
    df_map,
    locations="Country Name", locationmode="country names",
    color=map_metric, hover_name="Country Name", hover_data=["GDP per capita"],
    color_continuous_scale="YlGnBu",
    title=f"{map_metric} in {map_year}"
)
st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# 6. GDP Growth vs Poverty Change (2000–2020)
# =========================================================
st.header("6️⃣ Long-Term Effect – Growth vs Poverty Reduction")
st.markdown("*🔵 Bubble size = GDP per capita growth between 2000 and 2020. Larger bubbles indicate stronger economic performance.*")
st.markdown("""
#### Growth vs. Inclusion: Who Benefited?
This chart compares **GDP growth** and **poverty rate changes** between 2000 and 2020.
It reveals how **strong, inclusive growth** can drastically reduce poverty — and where it failed to do so.
""")

df_compare = df[df["Year"].isin([2000, 2020])]
df_wide = df_compare.pivot(index=["Country Name"], columns="Year", values=["GDP per capita", "Poverty rate"])
df_wide.columns = [f"{var}_{year}" for var, year in df_wide.columns]
df_wide = df_wide.dropna()

df_wide["GDP Growth (%)"] = ((df_wide["GDP per capita_2020"] - df_wide["GDP per capita_2000"]) / df_wide["GDP per capita_2000"]) * 100
df_wide["Poverty Change (pts)"] = df_wide["Poverty rate_2020"] - df_wide["Poverty rate_2000"]

fig_scatter = px.scatter(
    data_frame=df_wide,
    x="GDP Growth (%)", y="Poverty Change (pts)",
    size="GDP Growth (%)",
    hover_name=df_wide.index,
    color="Poverty Change (pts)",
    color_continuous_scale="RdBu",
    title="2000–2020 – GDP Growth vs Poverty Change",
    height=600,
    size_max=60
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("""
### Conclusion
This exploration reveals that **economic growth**, while powerful, is not a silver bullet.

Some key takeaways:
- Growth must be **inclusive** and accompanied by **sectoral transformation** to effectively reduce poverty.
- **Inequality** can undermine the benefits of growth if the gains are not broadly shared.
- The success of countries like **China** and **India** lies in their ability to leverage growth into real human development.

As we move forward, understanding these dynamics is essential to **design policies that ensure no one is left behind**.
""")


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
To understand how countries combat poverty, it is interesting to look at the financial lifelines flowing across borders like:           
- **Net Official Development Assistance (ODA):** Grants or concessional loans from government to support development.
- **Foreign Direct Investment (FDI):** Cross-border investments, indicating private sector engagement.
- **Personal Remittances:** Funds sent by individuals (often migrant workers) to their home countries.
.""")

st.altair_chart(world_financial_aid)
st.markdown(
'''
#### Key Observations:
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
- **Caribbean & Central America (2010):**  A dramatic spike in **Humanitarian Aid** corresponds with the devastating earthquake in **Haiti** in January 2010. The international community responded with unprecedented humanitarian assistance to support emergency relief and reconstruction efforts.    
- **Middle East:** Periodic increases in **Humanitarian Aid** can be linked to conflicts and refugee crises in the region, such as the Syrian conflict starting in 2011.
- **Europe**: Surge in aid flows in 2022 to support Ukraine in the conflict.

*This chart emphasizes how aid flows respond dynamically to regional crises, geopolitical events, and development priorities.*
'''
)

st.markdown("In this section, we dive deeper into the distribution of Official Development Assistance (ODA). We display both relative and absolute ODA flows to highlight the disparities in aid distribution.")

tab1, tab2 = st.tabs(["Absolute ODA", "Relative ODA"])
tab1.altair_chart(top_donor_recipient_chart, use_container_width=True)
tab2.altair_chart(top_donor_recipient_chart_bis, use_container_width=True)

st.markdown(""" 
            Even though official aid has grown over the years, it still falls significantly short of the 0.7% of GNI benchmark expected from developed nations to support less wealthy countries, 
            as outlined in SDG target 17.2. Only 4 countries achieved this goal in 2022. Overall the 5 larger donators in terms of absolute ODA are the same for decades: USA, Germany, France, UK and Japan.
            However, the relative ODA (as % of GNI) shows a different picture. The top 5 countries in terms of relative ODA are: Sweden, Norway, Luxembourg, Denmark and the UK.
            """)

st.subheader("Section 2: Bilateral ODA Flows Between Regions")
st.markdown("""
            The chord diagrams below visualize the bilateral ODA flows between regions in 1990 and 2021. The width of the chords represents the volume of aid flows,
             with wider chords indicating larger amounts of aid. It illustrates the evolution of aid distribution over time and highlights the regions that are the largest donors and recipients of ODA.
            The diagrams also show how the global landscape of aid flows has changed over the years, reflecting shifts in geopolitical priorities and development needs.
            """)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_1990)
with col2:
    st.pyplot(fig_2021)

st.markdown(
    """
#### Observations on Aid Distribution
When reading the diagrams, keep in mind that the overall increase in aid flows means chord sizes are not directly comparable across years.
East Asia & Pacific has emerged as a growing donor region, though almost half of its aid still stays within its own borders, highlighting persistent regional disparities.
Europe remains by far the largest donor, and its aid flows have become more diversified in 2021 compared to 1990, when they were heavily directed toward Sub-Saharan Africa
— a legacy partly rooted in colonial ties. Notably, Europe has doubled its aid to the Middle East, reflecting growing needs driven by conflicts and humanitarian crises.
 Meanwhile, North America has significantly expanded its role, more than tripling its aid to Sub-Saharan Africa, indicating a stronger engagement with the region’s development and emergency needs.
"""
)
st.markdown("""
            #### Deeper insights of bilateral ODA flows from donors to recipients at the country level in 2021
            **Click** on a pie segment to select a donor or recipient
""")
st.altair_chart(bilateral_oda, use_container_width=True)
st.markdown("""
### 🌍 Bilateral Official Development Assistance (ODA) Flows (2021)

This visualization provides an interactive view of bilateral ODA at the country level for the year 2021, measured in million USD.

The chart is divided into two main sections:

- **Left Side:**  
  - Displays the main **donors** that provide aid to the selected recipient country. You can click on the right diagram to select the recipient you want.

- **Right Side:**  
  - Displays the primary **recipients** of aid from the selected donor country. You can click on the left diagram to select the donor you want.

We can identify new patterns in the aid distribution, since some countries aid distribution sometimes comes from very few countries while in other case the origin of ODA are much more diverse.           
""")


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

scatter_poverty_oda = alt.Chart(poverty[poverty['Region'].notna()]).mark_point().transform_filter(
    year_selection2
).encode(
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
    brush # Apply the year selection filter
).properties(
    height=400,
    width=400,
    title="Poverty Headcount Ratio at $2.15/day vs ODA per Capita"
)

bar_oda_plot = alt.Chart(poverty[poverty['Region'].notna()]).mark_bar().transform_filter(
    year_selection2, brush
).encode(
    y=alt.Y('Country Name:N',
            sort='-x'),  # Sort by the x-axis values in descending order
    x="Net official development assistance received (constant 2021 US$):Q",
    color=alt.Color('Income Group:N',
                    scale=color_scale,
                    legend=alt.Legend(title="Income Group")),
    tooltip=['Country Name:N', 'Year:O', 'Net official development assistance received (constant 2021 US$):Q']
).transform_filter(brush).properties(
    height=500,
    width=300,
)

last_chart = (scatter_poverty_oda | bar_oda_plot).add_params(year_selection2)

col1, col2 = st.columns(2)
st.altair_chart(last_chart)

