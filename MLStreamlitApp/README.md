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

This makes the app feel less like a model generator and more like a guided machine learning learning space.

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

You can adjust key settings and immediately see how they affect results:

- Apply **feature scaling** when models depend on distance  
- Control **decision tree depth** to manage overfitting  
- Change **KNN neighbors (k)** to shift how predictions are made  
- Optionally run **GridSearchCV** to automatically find better parameters  

This turns tuning into something you can *see*, not just guess.
<img width="287" height="239" alt="Screenshot 2026-04-14 at 10 57 10 PM" src="https://github.com/user-attachments/assets/5a3ace63-ddb4-44a5-b93f-73b1bb979e07" />

---

### 📊 Clear, Visual Results

Model performance is presented in a way that is easy to interpret:

- Classification: accuracy, precision, recall, F1, confusion matrix, ROC curve  
- Regression: mean squared error, R², and predicted vs. actual plots  
- A comparison table lets you evaluate models side by side  

The focus is on understanding differences between models.

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
👉 https://gutierrez-data-science-portfolio-qlzv4ghbghhffhfxolraqp.streamlit.app/

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

