 # 📊 Federal RD Spending & GDP Analysis

## Overview

This project analyzes United States federal Research and Development spending across government agencies and compares it to GDP over time.
My goals where to understand:
- How RD spending is distributed across departments  
- Whether spending is increasing in meaningful ways  
- How RD growth compares to overall economic growth  

## Dataset: 

This project uses the **Federal R&D Budgets dataset** from TidyTuesday:  
https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12  
- `fed_rd_year&gdp.csv`

The dataset reports annual U.S. federal RD spending by department alongside GDP, allowing for both agency level comparisons and economy wide context.

### Pre-processing

The original data was structured in a wide format, with each department as its own column. I reshaped it into tidy form using `pandas.melt()`, creating a dataset where each row represents a single department’s spending in a given year. I then standardized variables (`year`, `department`, `rd_spending`, `gdp`), cleaned numeric formatting, and generated key measures including RD as a share of GDP and growth rates.

These steps make the data easier to compare across departments and over time, while allowing for more meaningful interpretation beyond just spending totals.

## Steps
### 1. Data Cleaning (Tidy Data)
The dataset was originally in **wide format** (each department as a column).
I transformed it into **tidy format** using:
- `pd.melt()` → reshaped data into long format  
- Created columns:
  - `year`
  - `department`
  - `rd_spending`
  - `gdp`

### 2. Feature Creation
Added key variables for analysis:
- RD as a percentage of GDP  
- Growth rates for RD and GDP  
- Spending in billions for readability  

### 3. Analysis & Visualizations
I created:
- Line graphs (RD by department over time)  
- Top 5 department comparisons  
- Growth comparisons (RD vs GDP)  

## Key Insights

- **RD spending is highly concentrated** in a few departments  
- The same departments dominate consistently (shows long-term priorities)
- **Spending increases does not mean it is a higher priority**  
  - GDP often grows just as fast or faster  
- Relative measures (Percentage of GDP) provide better insight than just dollars  

## How to Run
1. install pandas, matplotlib, and seaborn
2. Upload: fed_rd_year&gdp.csv
3. Run: main-1.ipynb


# References: 
- [Pandas Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Tidy Data Paper by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Data to Viz](https://www.data-to-viz.com/)





