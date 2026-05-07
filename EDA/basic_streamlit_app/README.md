# 🐧 Penguin Insights Dashboard 

This Streamlit dashboard explores physical differences among Adelie, Chinstrap, and Gentoo penguins. The app allows users to filter the dataset by species, island, and sex, then view how those choices change the visual results.

<img width="663" height="105" alt="Screenshot 2026-04-20 at 7 19 03 PM" src="https://github.com/user-attachments/assets/3ff087a3-a108-4b56-ba77-fdb70f72150d" />

## 📁 Purpose of Project  

- The purpose of this project is to build an interactive EDA tool that transforms raw biological measurements into visual insights.
- This helps users quickly compare species and analyze trait relationships.

## 📚 Table of Contents 

- Page 1: Exploring Our Dataset
     - This page gives users a broad overview of the filtered dataset. It includes summary metrics and a species count chart.
- Page 2: Body Variation
     - This page compares average body mass and flipper length across species and sex. It helps show how physical traits differ among penguin groups.
- Page 3: Key Bodily Correlations
     -  This page allows users to choose two numeric traits and compare them in a scatterplot. This makes it easier to see relationships between features such as flipper length and body mass.


## ⚙️ Features 
- Interactive Filtering
     - Filter penguins by species, island, and sex
     - Instantly updates all visualizations
- Descriptive Analytics
     - Key metrics (total count, species diversity, average body mass)
     - Aggregated comparisons across species and sex
- Data Visualizations
     - Bar charts for distribution and group comparisons
     - Scatter plots for exploring relationships between physical traits
- User Engagement
     - Interactive elements (fun fact button)
     - Intuitive layout for ease of use
 
<img width="750" height="593" alt="Screenshot 2026-04-20 at 7 19 38 PM" src="https://github.com/user-attachments/assets/14908634-bd1f-4cdd-a541-357a69d638ae" />

<img width="753" height="636" alt="Screenshot 2026-04-20 at 7 20 31 PM" src="https://github.com/user-attachments/assets/8fc0c0f0-7d26-4979-86e9-95de7dd0de7c" />


## 🔑 Key Insights
- Species Differences
       - Gentoo penguins exhibit the highest average body mass and longest flipper length.
- Sex-Based Variation
       - Male penguins tend to have greater body mass than females across species.
- Correlations
       - Strong relationships exist between body mass, flipper length, and bill dimensions.
  
<img width="774" height="610" alt="Screenshot 2026-04-20 at 7 20 09 PM" src="https://github.com/user-attachments/assets/3231ab0e-e427-4f76-83ab-9567a924ba80" />

## 🏃‍♀️ How to Run the App  

- Make sure to have Streamlit installed: 
    - $ pip install streamlit
- Streamlit run main.py

## 📕 References  
- [Pandas Cheat Sheet (PDF)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Data to Viz](https://www.data-to-viz.com/)



