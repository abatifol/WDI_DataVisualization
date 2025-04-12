import pandas as pd


# Load and preprocess the data from the World Bank
data = pd.read_csv('../WDI_CSV_2025_01_28/WDICSV.csv')
countries = pd.read_csv('../WDI_CSV_2025_01_28/WDICountry.csv')
df = pd.melt(data, id_vars=['Country Name', 'Country Code', 'Indicator Name',
                'Indicator Code'], var_name='Year', value_name='Value')
df = df.merge(countries[['Country Code', 'Region',
                'Income Group']], on='Country Code', how='left')
df['Year'] = df['Year'].astype(int)


# Filter indicators and regions
indicators = [
    "Net ODA provided, to the least developed countries (current US$)",
    "Net ODA provided, total (% of GNI)",
    "Net ODA provided, total (constant 2021 US$)",
    "Net ODA provided, total (current US$)",
    "Net ODA received (% of central government expense)",
    "Net ODA received (% of GNI)",
    "Net ODA received per capita (current US$)",
    "Net official development assistance received (constant 2021 US$)",
    "Net official development assistance received (current US$)",
    "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)",
    "Population, total",
    "Personal remittances, received (current US$)",
    "Foreign direct investment, net inflows (BoP, current US$)",
    "GNI (current US$)","GDP per capita (current US$)",
    "GDP (current US$)"
]
df = df[df['Indicator Name'].isin(indicators)]

# # Pivot and clean
df = pd.pivot_table(df, 
                    index=['Country Name','Country Code', 'Year'],
                    columns='Indicator Name', 
                    values='Value').reset_index()

df.columns = [col.replace(' (2017 PPP) (% of population)', '') for col in df.columns]
df['Year'] = df['Year'].astype(int)
df = df[df.Year >=1990]
df.to_parquet('./data/oda_poverty_world.parquet', engine='pyarrow', index=False)

countries = pd.read_csv('../WDI_CSV_2025_01_28/WDICountry.csv')
countries  = countries[['Country Code','Region','Income Group']]
countries.to_parquet('./data/WDICountry.parquet', engine='pyarrow',index=False)


# Load and preprocess the data from the OECD
oda = pd.read_csv('../data_csv/ODA_disbursment.csv')
countries = pd.read_csv('../WDI_CSV_2025_01_28/WDICountry.csv')
correspondance = pd.read_csv('../data_csv/Correspondences_DAC2a.csv')
oda  = oda[(oda.Year >= 1990) & (oda.Year <= 2022) & (oda['Amount type']=='Constant Prices (2022 USD millions)')]
countries = countries.rename(columns={'Country Code':'DE_code'})
correspondance = correspondance.merge(countries[['DE_code','Region']]).drop(columns='DE_code')
correspondance['dotstat_code']=correspondance['dotstat_code'].astype(int)
oda = oda.merge(correspondance.rename(columns={'dotstat_code':'RECIPIENT','Region':'Recipient_Region'}), how='left')
oda = oda.merge(correspondance.rename(columns={'dotstat_code':'DONOR','Region':'Donor_Region'}), how='left')
oda = oda[oda['Aid type'].isin(['Imputed Multilateral ODA', 'ODA: Total Net', 'Technical Cooperation','Memo: ODA Total, Gross disbursements',
       'ODA as % GNI (Recipient)','ODA per Capita', 'Development Food Aid', 'Humanitarian Aid'])]
oda.to_parquet('./data/oda_disbursment.parquet', engine='pyarrow', index=False)


# prepare donor and recipient dataframes to display top 10 recipients/donor and group the rest in Other for each donor/recipient
oda_filtered = oda[oda.Recipient_Region.notna() & oda.Donor_Region.notna()].copy()
donor_recipient = (
    oda_filtered
    .groupby(['Recipient', 'Donor'], as_index=False)['Value']
    .sum()
)
donor_recipient['Rank'] = donor_recipient.groupby('Donor')['Value'].rank(method='first', ascending=False)
donor_recipient['Recipient_Group'] = donor_recipient['Recipient'].where(donor_recipient['Rank'] <= 10, 'Others')
df_prepared = (
    donor_recipient
    .groupby(['Donor', 'Recipient_Group'], as_index=False)['Value']
    .sum()
)
donor_prepared = df_prepared.rename(columns={'Recipient_Group':'Recipient'})
donor_prepared.to_parquet("./data/chart8_donor_prepared.parquet",engine="pyarrow", index=False)

oda_filtered = oda[oda.Recipient_Region.notna() & oda.Donor_Region.notna()].copy()
donor_recipient = (
    oda_filtered
    .groupby(['Recipient', 'Donor'], as_index=False)['Value']
    .sum()
)
donor_recipient['Rank'] = donor_recipient.groupby('Recipient')['Value'].rank(method='first', ascending=False)
donor_recipient['Donor_Group'] = donor_recipient['Donor'].where(donor_recipient['Rank'] <= 10, 'Others')
df_prepared = (
    donor_recipient
    .groupby(['Recipient', 'Donor_Group'], as_index=False)['Value']
    .sum()
)
recipient_prepared = df_prepared.rename(columns={'Donor_Group':'Donor'})
recipient_prepared.to_parquet("./data/chart8_recipient_prepared.parquet",engine="pyarrow", index=False)