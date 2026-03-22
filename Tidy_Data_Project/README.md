 # 📊 Federal RD Spending & GDP Analysis

## Overview

This project analyzes United States federal Research and Development spending across government agencies and compares it to GDP over time.
My goals where to understand:
- How RD spending is distributed across departments  
- Whether spending is increasing in meaningful ways  
- How RD growth compares to overall economic growth  

## Dataset: 
`fed_rd_year&gdp.csv`

Includes:
- Yearly RD spending by department  
- GDP values  

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
2. Run: main-1.ipynb

# References: 
- [Tidy Data Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Tidy Data Paper by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Visualization Choices](https://www.data-to-viz.com/)
- [Visualization Colors](https://xkcd.com/color/rgb/)




