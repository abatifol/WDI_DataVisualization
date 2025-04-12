# 🌍 Interactive Dashboard for International Aid & Economic Development

## Project Overview

This project is part of the Data Visualization (2025) class in CentraleSupélec. It explores **SDG 1: No Poverty** and **SDG 17: Partnership for the goals** by visualizing the relationship between **economic growth**, **international aid flows** and poverty. Using data from the **World Bank** and **OECD**, we built an **interactive dashboard** that helps users understand how different countries and regions contribute to global poverty reduction through **Official Development Assistance (ODA)**.

It is built with **Streamlit**, **Altair**, and **Plotly** for dynamic visualizations.

## Data Sources

- **World Bank — World Development Indicators (WDI):**  
  *https://datacatalog.worldbank.org/search/dataset/0037712*

- **OECD — DAC2A Official Development Assistance (ODA) flow:**  
  *https://data-explorer.oecd.org/vis?lc=en&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_DAC2%40DF_DAC2A&df[ag]=OECD.DCD.FSD&df[vs]=1.3&dq=....&lom=LASTNPERIODS&lo=5&to[TIME_PERIOD]=false*


## Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/abatifol/WDI_DataVisualization.git
cd WDI_DataVisualization
```

### 2. Install Dependencies

Make sure you have Python installed and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

Your browser will open automatically with the dashboard!


## Contributors

- **Antonine Batifol**
- **William Slimi**


## Limitations

- Current data may not include the latest data since there is no real-time update.
