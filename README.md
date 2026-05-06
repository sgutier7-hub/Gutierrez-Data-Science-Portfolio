# Data Science Portfolio 📊 

## Repository Purpose 
This repository showcases my data science projects, highlighting my technical skills in data analysis, visualization, and machine learning.  

Each project reflects my ability to transform data into meaningful insights and build tools that make analysis more accessible.

---

## Project Table of Contents 📂

Projects in this portfolio are organized into the following sections:

- **[EDA](./EDA/)** — Exploratory data analysis

  - [Penguins Insights Dashboard 🐧](./basic_streamlit_app/README.md)
      - An interactive Streamlit app for exploring the Palmer Penguins dataset through  filters and comparative visualizations.

      **What it does:**  
      - Allows users to explore species differences 
      - Visualizes body variation and key trait relationships  
      - Highlights correlations between physical measurements  

      **Why this project matters:**  
      - Demonstrates ability to build **interactive data tools**
      - Shows understanding of **EDA, visualization design, and user experience**

      This project complements my portfolio by adding an interactive component, showing that I can design tools that make data exploration more accessible.

---

- **[Visualizations](./Visualizations/)** — Data visualization and storytelling
    
  - [Federal RD Spending & GDP Analysis 💵](./Tidy_Data_Project/README.md)  
    - This project transforms and analyzes U.S. federal RD spending data. I convert a wide format dataset into tidy form using pandas, then perform exploratory data analysis to examine how RD investment is distributed across departments and how it changes relative to GDP over time.

    **Why this project matters:**  
    - Structures real-world data into a usable format  
    - Applies tidy data principles 
    - Moves beyond visualization to interpret results  

    This project complements my broader portfolio by combining data cleaning and analytical thinking, which aligns with my interests in law and public policy.

---

- **[Models](./Models/)** — Statistical and machine learning models

  - [Interactive ML Model Explorer 💡](./MLStreamlitApp/README.md)  
    - An interactive Streamlit app that walks users through the full machine learning workflow 

    **What it does:**  
    - Lets users upload their own dataset and choose a target variable  
    - Automatically suggests classification or regression (with option to override)  
    - Recommends models based on the problem type  
    - Allows users to adjust key settings (scaling, tree depth, KNN neighbors)  
    - Displays results through metrics, confusion matrices, ROC curves, and regression plots  
    - Compares multiple models side-by-side  

    **Why this project matters:**  
    - Moves beyond static modeling to create a **fully interactive ML experience**  
    - Shows understanding of the **full ML pipeline** (data prep to model selection to evaluation)  
    - Emphasizes **interpretability and decision-making**, not just outputs  
    - Demonstrates ability to design tools that make technical concepts more accessible  

    This project is one of the most comprehensive pieces in my portfolio. It brings together data exploration, modeling, and user experience into a single tool, showing not just that I can build models, but that I understand how and why they work.

  - [Uncovering the Unseen: Unsupervised Learning in Practice 🔎](./MLUnsupervisedApp./README.md)  
    - An interactive Streamlit application that explores dimensionality reduction and clustering through Principal Component Analysis (PCA), k-means clustering, and hierarchical clustering  

    **What it does:**  
    - Allows users to upload their own CSV datasets or experiment with built-in datasets 
    - Automatically preprocesses messy real-world data by converting categorical, boolean, and datetime variables into usable numeric features for modeling  
    - Cleans and standardizes the dataset so PCA and clustering algorithms compare features based on structure rather than raw scale differences  
    - Uses Principal Component Analysis (PCA) to compress high-dimensional datasets into lower-dimensional representations that preserve the strongest variation patterns  
    - Visualizes PCA through 2D projections, loading value tables, variance explained analysis, and cumulative variance elbow charts  
    - Uses k-means clustering with interactive cluster selection, elbow method evaluation, and silhouette score analysis  
    - Uses hierarchical clustering using ward linkage to generate nested cluster structure and dendrogram visualizations  
    - Projects clustering assignments onto PCA scatterplots to make latent group structure easier to interpret  
    - Includes written interpretation throughout the application so users can understand what the models are actually detecting inside the data rather than only viewing outputs  

    **Why this project matters:**  
    - Demonstrates understanding of unsupervised learning beyond running prebuilt models by emphasizing preprocessing decisions, variance structure, clustering evaluation, and interpretation  
    - Highlights technical knowledge of dimensionality reduction, distance-based clustering, hierarchical clustering behavior, variance analysis, and cluster quality metrics  
    - Emphasizes interpretability by connecting mathematical concepts like PCA loadings, silhouette scores, and dendrogram structure to visual explanations users can actually explore    

    This project became much more than a clustering app for me. I wanted to build something that exposed how unsupervised learning actually behaves beneath the surface instead of treating PCA and clustering like black-box functions that automatically generate insight. A lot of machine learning projects stop at producing charts or labels, but this application focuses on helping users understand *why* certain structure appears and how preprocessing changes the analysis. Building this project forced me to think carefully about the relationship between visualization and interpretation, especially how to make high-dimensional structure understandable without stripping away the complexity that makes these methods powerful in the first place.
