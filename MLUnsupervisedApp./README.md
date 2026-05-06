# 🔎 Uncovering the Unseen: Unsupervised Learning in Practice

Unsupervised learning can feel abstract when it is only presented through equations or static examples. I built this interactive Streamlit application to make unsupervised machine learning more visual, interpretable, and hands-on by allowing users to explore how patterns emerge inside real datasets.

This project combines Principal Component Analysis (PCA), K-Means clustering, and Hierarchical Clustering into a single workflow that guides users from raw uploaded data to interpretable machine learning results. Rather than functioning as a black-box modeling tool, the app was intentionally designed to explain how each method works, why certain preprocessing decisions matter, and how clustering results should be interpreted.

Users can upload their own CSV datasets or experiment with built-in datasets such as Titanic, Penguins, and Iris. The application automatically cleans and prepares the data for modeling, then provides interactive visualizations and evaluation tools that help users investigate latent structure, dimensionality reduction, and clustering behavior.

---

# Project Overview

Most datasets contain structure that is difficult to detect from tables or summary statistics alone. Patterns such as hidden groupings, overlap between observations, correlated variables, and high-dimensional relationships often require machine learning methods to uncover.

This application walks users through the full unsupervised learning workflow, including:

- dataset exploration
- preprocessing and feature transformation
- dimensionality reduction
- clustering analysis
- cluster evaluation
- visualization and interpretation

The app allows users to:

- upload CSV datasets or use built-in sample datasets
- inspect missing values and cleaned modeling features
- reduce dimensionality using PCA
- evaluate variance explained and loading values
- perform K-Means clustering
- perform Hierarchical Clustering using Ward linkage
- compare clustering quality using elbow plots and silhouette scores
- visualize clusters using PCA projections and dendrograms

The project emphasizes interpretation alongside modeling by pairing visual outputs with explanations of how each method behaves mathematically and how results should be analyzed.

---

## 🚀 How to Run the App

### 🌐 Try it online
You can use the app directly here — no setup needed:  
👉 https://gutierrez-data-science-portfolio-qlzv4ghbghhffhfxolraqp.streamlit.app/

---

### 💻 Run locally

1. Start by cloning this repository

```bash
git clone https://github.com/sgutier7-hub/Gutierrez-Data-Science-Portfolio.git
cd Gutierrez-Data-Science-Portfolio/MLUnsupervisedApp.
pip install -r requirements.txt
streamlit run MLUnsupervisedApp.py
```
3. Install dependencies
---

# Required Libraries and Versions

Create a `requirements.txt` file with:

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
scipy
```

---

# Sample Datasets Included

The app includes:

- Titanic Dataset
- Penguins Dataset
- Iris Dataset

Users can either upload their own CSV dataset or experiment with one of the built-in datasets.

---

# 🧹 Data Cleaning and Preprocessing

Before modeling begins, the app automatically preprocesses the dataset so PCA and clustering algorithms can operate on fully numeric input.

The preprocessing pipeline:

- converts boolean variables into 0/1 format
- detects datetime columns and converts them into timestamps
- factorizes categorical variables into numeric codes
- removes remaining non-numeric columns
- removes rows with missing values

The app also standardizes all modeling features using `StandardScaler`.

Standardization transforms features so they have:

- mean = 0
- standard deviation = 1

This step is essential because PCA and clustering are distance and variance based methods. Without standardization, variables with larger numeric scales would dominate the analysis and distort the clustering structure.

---

# Machine Learning Models Used

# 📕 Principal Component Analysis (PCA)

## Purpose

Principal Component Analysis (PCA) reduces high-dimensional datasets into a smaller number of principal components that capture the strongest variation in the data.

Instead of analyzing every feature independently, PCA creates new variables that summarize shared structure across the dataset:

- PC1 captures the greatest amount of variation
- PC2 captures the next greatest amount while remaining independent from PC1
- additional components capture progressively smaller sources of variation

This helps simplify large datasets while preserving the strongest overall patterns.

---

## Why PCA Was Used

PCA was included to:

- reduce redundancy between correlated variables
- reveal hidden structure in high-dimensional data
- identify overlap, separation, and outliers
- visualize complex datasets more clearly
- support clustering interpretation through lower-dimensional projections

---

## Features Included

The PCA section includes:

- 2D PCA projection scatterplots
- loading value tables
- variance explained tables
- cumulative variance elbow charts
- individual vs cumulative variance visualizations

---

## Hyperparameter Selection

### Number of Principal Components (`n_components`)

The primary PCA hyperparameter is:

```python
n_components
```

This determines how many principal components PCA retains.

Users select the number of components interactively with a slider. The maximum number of components is automatically constrained by:

- the number of cleaned features
- the number of observations

The app also provides:

- cumulative variance charts
- variance explained tables
- elbow-style variance plots

These tools help users determine how many components preserve meaningful structure before additional components begin contributing minimal new information.

---

# 📗 K-Means Clustering

## Purpose

K-Means groups observations into clusters based on feature similarity by assigning points to the nearest centroid and repeatedly updating cluster centers until the assignments stabilize.

---

## Why K-Means Was Used

K-Means was included because it is:

- computationally efficient
- easy to interpret
- effective for compact, well-separated clusters
- useful for exploring grouping structure interactively

The model also produces clear outputs such as:

- cluster assignments
- centroid positions
- within-cluster variation
- clustering evaluation metrics

---

## Hyperparameters

### Number of Clusters (`k`)

The primary K-Means hyperparameter is:

```python
k = number of clusters
```

This determines how many groups the dataset will be divided into.

Choosing k is important because:

- too few clusters may combine distinct observations
- too many clusters may split meaningful structure into artificial groups

The app allows users to compare multiple cluster counts interactively.

---

## How the App Helps Select k

### Elbow Method

The elbow method evaluates clustering performance using:

```python
Within-Cluster Sum of Squares (WCSS)
```

WCSS measures how closely observations fit around their assigned centroid.

Lower WCSS values indicate:

- tighter clusters
- lower within-cluster variation

The elbow occurs where additional clusters begin producing only small reductions in WCSS.

---

### Silhouette Scores

The app also calculates silhouette scores across multiple cluster counts.

The silhouette score measures how well observations fit within their assigned cluster compared to neighboring clusters.

Higher silhouette scores generally indicate:

- stronger cluster separation
- tighter grouping structure
- clearer boundaries between clusters

Using silhouette scores together with the elbow method provides a more balanced way to evaluate clustering quality.

---

## Additional Model Settings

### `random_state = 42`

A fixed random state ensures:

- reproducibility
- consistent clustering results
- easier comparison across runs

---

### `n_init = 20`

The application uses:

```python
n_init = 20
```

K-Means runs multiple initializations using different centroid starting points and selects the solution with the lowest within-cluster variation. This improves clustering stability and reduces the likelihood of poor local solutions.

---

# 📘 Hierarchical Clustering

## Purpose

Hierarchical clustering builds nested cluster structure by repeatedly merging the most similar observations together.

Unlike k-means, hierarchical clustering does not immediately assign observations into a fixed number of groups. Instead, it creates a full hierarchy that preserves the merge history of the dataset.

---

## Why Ward Linkage Was Used

This project specifically uses:

```python
linkage = "ward"
```

Ward linkage minimizes increases in within cluster variance during merging.

Compared to other linkage methods:

- **single linkage** can create chaining effects
- **complete linkage** is more sensitive to outliers
- **average linkage** does not directly minimize cluster variance

Ward linkage was selected because it generally produces:

- more compact clusters
- clearer group separation
- smoother hierarchical structure
- easier-to-read dendrograms

---

## Hyperparameter Selection

### Number of Clusters

T hierarchical clustering builds the full hierarchy automatically, but the app allows users to select the number of clusters interactively with a slider.

The application also computes silhouette scores across multiple cluster counts to help evaluate clustering quality and compare grouping structure.

---

## Features Included

The hierarchical clustering section includes:

- dendrogram visualizations
- cluster size tables
- silhouette score analysis
- PCA cluster scatterplots

---

# 📊 Visualizations Included

The application includes interactive visualizations for:

- missing value analysis
- feature relationships
- PCA projections and loading values
- cumulative variance analysis
- elbow method evaluation
- silhouette score comparison
- dendrogram structure
- PCA-based cluster visualization

---

# 📚 References

- [Principal Component Analysis](https://towardsdatascience.com/principal-component-analysis-fbce2a22c6e0/)
- [K-Means Clustering – Explained](https://towardsdatascience.com/k-means-clustering-explained-4528df86a120/)
- [Hierarchical Clustering Explained (Medium)](https://medium.com/data-science/hierarchical-clustering-explained-e59b13846da8/)
- [Hierarchical Clustering Explained (Towards Data Science)](https://towardsdatascience.com/hierarchical-clustering-explained-e58d2f936323/)
- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Machine Learning Algorithms Cheat Sheet](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms-cheat-sheet/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Professor Smiley's In-Class Coding Notebooks and Slides

