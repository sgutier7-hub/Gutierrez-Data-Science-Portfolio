# 💡 Data Insight with Interactive Machine Learning Models

## Project Overview

This project is an interactive Streamlit app designed to help you *actually understand* what’s happening inside your machine learning workflow; not just run models and hope for the best.

Most machine learning tools can feel like black boxes. You upload data and get results without really knowing why a model worked, why it failed, or what you should change. This app pushes back against that.

Instead of making machine learning feel distant or automatic, this app walks users through the process step by step. It encourages you to slow down and explore your dataset to make sense of the choices behind the model.

Throughout the app, you move through key parts of the workflow:
- examining the structure of your data
- choosing the variable you want to predict
- identifying whether your task is classification or regression
- seeing which models fit that type of problem
- testing models and comparing how they perform

### What You Can Do

- Upload your own CSV dataset
- Automatically detect the problem type, or choose it manually
- Explore the dataset before modeling
- View model recommendations based on your target variable
- Compare multiple machine learning models at the same time
- Adjust settings like scaling, decision tree depth, and KNN neighbors
- View results through metrics, confusion matrices, ROC curves, and regression plots

## App Features

This app is built to guide you through the machine learning process in a way that is interactive easy to follow. Each section is designed to help you understand not just *what* is happening, but *why*.

### 🔍 Explore Your Data First

Before any modeling happens, the app helps you understand your dataset:

- Preview your data and inspect its structure  
- Identify missing values and column types  
- Visualize your target variable (bar chart for classification, histogram for regression)  
- Explore relationships between variables with a correlation heatmap

<img width="704" height="385" alt="Screenshot 2026-04-14 at 10 51 38 PM" src="https://github.com/user-attachments/assets/bfc862a4-e975-46f5-8b27-c890e8ea51f8" />

---

### 🧠 Smart Problem Type Detection

The app suggests whether your task is:

- **Classification**: when your target is categorical or has a small number of unique values  
- **Regression**: when your target is continuous and numeric  

You can override this at any time, but the goal is to help you recognize the difference and why it matters.

<img width="288" height="92" alt="Screenshot 2026-04-14 at 10 53 56 PM" src="https://github.com/user-attachments/assets/aa7cc503-9ff4-4b37-9fbe-5a84884e438c" />

---

### 🎯 Model Selection That Makes Sense

Rather than overwhelming you with options, the app focuses on a small set of foundational models:

- **Logistic Regression**: a strong, interpretable starting point for classification  
- **Decision Tree**: captures non-linear patterns through simple decision rules  
- **KNN (K-Nearest Neighbors)**: classifies based on similarity between observations  
- **Linear Regression**: a baseline model for predicting continuous values  

Each model is chosen because it teaches something different about how machine learning works.

<img width="276" height="297" alt="Screenshot 2026-04-14 at 10 55 28 PM" src="https://github.com/user-attachments/assets/3906decc-9f86-4e54-a178-51d4637115f6" />

---
### 🖥️ Interactive Model Tuning

The app allows users to experiment with several important model settings and immediately observe how those choices affect performance.

Users can:
- apply feature scaling
- adjust decision tree depth
- change the number of neighbors used in KNN
- optionally use GridSearchCV for automated parameter tuning

This helps connect hyperparameter choices to real model behavior rather than treating tuning as trial and error.

<img width="287" height="239" alt="Screenshot 2026-04-14 at 10 57 10 PM" src="https://github.com/user-attachments/assets/5a3ace63-ddb4-44a5-b93f-73b1bb979e07" />

---

## ⚙️ Model Training and Hyperparameter Tuning

The app does more than just run machine learning models. It gives users control over important training decisions so they can see how those choices affect performance.

Before training begins, the dataset is cleaned and prepared for modeling:

- categorical variables are converted into numeric form when needed
- boolean values are converted into integers
- missing values are handled automatically depending on the workflow
- features can optionally be standardized using `StandardScaler`

The dataset is then split into training and testing sets so the model can be evaluated on unseen data rather than memorizing the original dataset.

### Adjustable Hyperparameters

Different models include different tuning options because each algorithm learns in a different way.

#### 🌳 Decision Tree
Users can control:

- **Maximum Depth (`max_depth`)**

This limits how deep the tree can grow.  
A shallow tree may underfit the data, while a very deep tree can memorize the training set and overfit.

---

#### 👥 K-Nearest Neighbors (KNN)
Users can control:

- **Number of Neighbors (`k`)**

This changes how many nearby observations the model uses when making predictions.

- Smaller values of `k` make the model more sensitive to local patterns and noise
- Larger values smooth predictions but may overlook important structure

Because KNN is distance-based, scaling is especially important.

---

#### 📏 Feature Scaling

Users can choose whether to standardize features before training.

Scaling transforms features so they are measured on comparable ranges.  
This matters for algorithms like:

- KNN
- Logistic Regression

Without scaling, variables with larger numeric ranges can dominate the model.

---

#### 🔍 GridSearchCV

The app also includes optional hyperparameter optimization using `GridSearchCV`.

This tests multiple parameter combinations automatically and selects the configuration that performs best on the training data.

Rather than manually guessing settings, users can compare how different hyperparameter choices influence model performance.

---

## 📈 Model Outputs and Evaluation

After training, the app generates multiple outputs to help users interpret model behavior and performance.

### Classification Outputs

For classification models, the app displays:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve and AUC score

These metrics help show not just whether the model is correct, but *how* it is making mistakes.

For example:

- precision focuses on false positives
- recall focuses on false negatives
- the confusion matrix shows where predictions are being misclassified

---

### Regression Outputs

For regression models, the app displays:

- Mean Squared Error (MSE)
- R² score
- Predicted vs. Actual scatterplots

These outputs help users evaluate overall model fit and prediction accuracy.

---

### 📊 Model Comparison

When multiple models are selected, the app creates a comparison table so users can evaluate performance side by side.

This makes it easier to recognize that different models may perform well for different reasons rather than assuming there is always one universally “best” model.

<img width="777" height="716" alt="Screenshot 2026-04-14 at 11 02 22 PM" src="https://github.com/user-attachments/assets/91432729-b12f-40b8-baeb-dd7dba117a0e" />

---

### ⚠️ Built-In Guardrails

The app also helps you catch common issues:

- Incompatible model + data combinations  
- Too few classes for classification  
- Missing values affecting performance  
- Situations where scaling matters

If something doesn’t run, it’s usually pointing you toward a data or modeling decision worth revisiting.

## ✅ Required Libraries
```bash
streamlit
pandas
numpy
scikit-learn
matplotlib
```

## 🚀 How to Run the App

### 🌐 Try it online
You can use the app directly here — no setup needed:  
👉 https://gutierrez-data-science-portfolio-ve2gve3rmvkxlfa5sih9xg.streamlit.app/

---

### 💻 Run locally

1. Start by cloning this repository

```bash
git clone https://github.com/sgutier7-hub/Gutierrez-Data-Science-Portfolio.git
cd Gutierrez-Data-Science-Portfolio/MLStreamlitApp
pip install -r requirements.txt
streamlit run app.py
```
3. Install dependencies
# References: 
- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Machine Learning Algorithms Cheat Sheet](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms-cheat-sheet/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Professor Smiley's In-Class Coding Notebooks and Slides

