
# SETTING UP MY CODE 

# Obviously need streamlit to build the app
import streamlit as st
# Pandas is used to clean and load the data
import pandas as pd
# Numpy is for the calculations
import numpy as np
# Matplotlib is for making the visualizations
import matplotlib.pyplot as plt

# I will need to split up my data into training and testing groups, so this is for that
# GridSearchCV hel[s find the best hyperparameters
from sklearn.model_selection import train_test_split, GridSearchCV

# This scales the features so they are in the same range 
from sklearn.preprocessing import StandardScaler

# Logistic Regression is a classification model for binary categories
# Linear Regression is for predicting continous numbers 
from sklearn.linear_model import LogisticRegression, LinearRegression

# The tree splits the data into branches to classify the decisions
from sklearn.tree import DecisionTreeClassifier

# KNN classifies points based on the closest data points to it
from sklearn.neighbors import KNeighborsClassifier

# I will use these to check how good the models are for a particular dataset
# Accuracy, precision, recall, and f1 are for classification
# auc is the score of the ROC curve (perfect models have a score of 1, the closer to 0 the worse the model is)
# The classification report shows the accuracy, precision, recall, and f1 score together
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, mean_squared_error, r2_score)

st.title("Data Insight with Interactive Machine Learning Models💡")

# Saves the dataset which makes the app run faster
@st.cache_data

# This function is to load the upploaded file
# It reads the file and makes it a dataframe
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# This function decides whether the problem should be classification or regression
# THIS IS NOT PERFECT! Obviously there are some cases that these recommendations won't match
def decide_problem_type(y):
# If there is text it will recommend a classification model
    if y.dtype == "object":
        return "Classification"
# If its numeric but doesn't many unique values it will also recommend a classification model
    if y.nunique(dropna=True) <= 10:
        return "Classification"
# Otherwise it'll recommend regression
    return "Regression"

# This is for dealing with missing data
# Users can drop missing values or keep the missing data in 
# .copy() will prevent the original dataset from being changed
# FOR SOME OF THE MODELS, YOU CANNOT HAVE MISSING DATA, BUT THE USER CAN GO BACK AND REMOVE MISSING DATA TO USE THAT MODEL
def clean_dataset(df, cleaning_method):
    return df.dropna().copy() if cleaning_method == "Drop rows with missing values" else df.copy()

# Prep dataset for ML model
# Use copy() to not change the original data
def prepare_features(df, target):
    model_df = df.copy()
# this finds the columns that have text and coverts them to numbers
    for col in model_df.select_dtypes(include="object").columns:
        model_df[col] = pd.factorize(model_df[col])[0]
# X is the variable to make predictions, Y is the variable we are trying to predict
    X = model_df.drop(columns=[target])
    y = model_df[target]
# I use the return function to get the features and target seperately to use them later on
    return X, y

# This function recommends a specific model based on the problem type
# I have a list of classification models for my app 
# If the dataset is binary, I will recommend a logistic regression
# If its not a classification problem, I will recommend a a linear regression 
def recommend_models(problem_type, y):
    if problem_type == "Classification":
        recommendations = [("Decision Tree (Classifier)", "Useful when the relationship is not purely linear and when feature combinations matter."),
            ("KNN (Classifier)", "Can work well when similar observations tend to belong to the same class, especially after scaling.")]
        if y.nunique() == 2:
            recommendations.insert(0, ("Logistic Regression", "A strong starting model for binary classification because it is simple, interpretable, and was used in your notebooks."))
        return recommendations
    return [("Linear Regression", "The regression model directly used in your notebooks and a strong baseline when relationships are fairly direct.")]

# This function scales the data 
def scale_data(X_train, X_test, do_scaling):
# If the user decides not to scale then the original training and testing data is used unchange represented by "None"
    if not do_scaling:
        return X_train.copy(), X_test.copy(), None
# Scale the training data, then convert it back into a DataFrame (because the StandardScaller has no column names)
# So we keep the original column names and row labels
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index)
# Apply the SAME scaling (from training data) to the test data
# # Do NOT fit again; this keeps scaling consistent 
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler

# SETUP IS DONE

# This is to give an overview of what my app will do 
# The st.expander becomes a dropdown that users can click
# Triple quotes lets me write all the whole list without breaking the code
with st.expander("What this app does"):
    st.write("""
    - Upload a CSV file
    - Choose a target variable
    - Show all model options
    - Explain which models fit the selected problem type
    - Change decision tree and KNN settings
    - See confusion matrices, classification reports, ROC curves, and regression metrics
    - Compare performance with and without scaling when that matters
    """)

# MOVING ONTO MAKING THE SIDEBAR

# The file uploader lets users upload a csv file, I specified the tupe
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# If there is no file then I will show a message so that the user knows they need one
# st.stop prevents the app working without a file
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# The CSV file will go into a datafram
df = load_data(uploaded_file)

# Chosing the target is what we want to predict
st.sidebar.header("2. Choose the Target")
# This lets the user chose the column that is the target 
target = st.sidebar.selectbox("Select the target variable", df.columns)

# This defines the possible problem types
problem_types = ["Classification", "Regression"]
# This automatically sugeests a problem type based on the target column
auto_problem_type = decide_problem_type(df[target])
# This lets the user CHANGE the problem type if they want
problem_type = st.sidebar.radio(
    "Problem type",
    problem_types,
    index=problem_types.index(auto_problem_type))

st.sidebar.header("3. Cleaning and Split")
# This gives the user the option on what to do with missing values
# HOWEVER SOME MODELS WON"T WORK WITH MISSING DATA, but they can go back and change this part 
cleaning_method = st.sidebar.selectbox(
    "Missing value handling",
    ["Drop rows with missing values", "Keep data as is"])
# This limits the amount of data that can be used for testing
# The range is from .1-.9 (random), it is set to .2 because that is the common setting
test_size = st.sidebar.slider("Test set size", 0.10, 0.90, 0.20, 0.05)
# The random state is set to a number so that the split is consistent in the results
# I set this at 42 because that is the common number
random_state = st.sidebar.number_input("Random state", 0, 9999, 42)

# This applies the selected cleaning method to the dataset
filtered_df = clean_dataset(df, cleaning_method)
# This splits the data into features and target for the modeling
X, y = prepare_features(filtered_df, target)

# SOME ERROR PREVENTION

# Regression models require a numeric target variable
if problem_type == "Regression":
    if not pd.api.types.is_numeric_dtype(y):
        st.error("Regression requires a numeric target.")
        st.stop()
# Classification requires at least two classes to make predictions
if problem_type == "Classification":
    if y.nunique(dropna=True) < 2:
        st.error("Classification requires at least two target classes.")
        st.stop()

# CHOOSING A MODEL 

st.sidebar.header("4. Choose Models")

# I am using four different models for my app 
available_models = ["Logistic Regression",
    "Decision Tree (Classifier)",
    "KNN (Classifier)",
    "Linear Regression"]

# This sets a model based on the problem type
if problem_type == "Classification":
    default_models = ["Logistic Regression", "Decision Tree (Classifier)"]
else:
    default_models = ["Linear Regression"]

# User can select multiple models to compare how they do 
model_choices = st.sidebar.multiselect("Select models to compare",
    available_models,
    default=default_models)

# HYPERPARAMETERS

st.sidebar.header("5. Change Settings")
# Gives the option to scale features, the value is set to valse so that user can choose to turn it on
do_scaling = st.sidebar.checkbox("Apply StandardScaler", value=False)
# Change the depth of the decision tree (can help with overfitting and getting big of a tree)
# 1 is the minimum value, 20 is the max, and 5 is the starting point 
max_depth = st.sidebar.slider("Decision Tree max depth", 1, 20, 5)
# Choose amount of neighbors for KNN
k = st.sidebar.slider("KNN neighbors (k)", 1, 20, 5,)

# Gives option to use GridSearchCV for the decision tree
# GridSearchCV finds the best parameters
# It will only show if the decision tree is selected
run_grid_search = (
    st.sidebar.checkbox("Run Decision Tree GridSearchCV")
    if problem_type == "Classification" and "Decision Tree (Classifier)" in model_choices
    else False)


# CURRENT SETUP

# Shows the settings the user has currently
st.header("Current Setup")
st.write("**Target:**", target)
st.write("**Problem type:**", problem_type)
st.write("**Rows:**", len(X))
st.write("**Features:**", X.shape[1])
st.write("**Scaling:**", do_scaling)
st.write("**Test size:**", test_size)


# TRAIN AND TEST SPLITITNG

# Decide whether stratified splitting can be used
# Stratification keeps proportions the same in train and test sets
# This only works if each class has at least 2 examples
use_stratify = (problem_type == "Classification" and y.value_counts().min() >= 2)
stratify_value = y if use_stratify else None

# This tells the user if stratification cannot be applied
if problem_type == "Classification" and not use_stratify:
    st.warning("Stratified splitting was turned off because at least one class has fewer than 2 examples.")

# Split the dataset into training and testing sets
# X is the features (inputs) and y is the target (what we predict)
# The test_size and rndom_state are set to the set amounts from earlier
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=stratify_value)

# This applies the scaling to the data if selected
# It returns scaled versions of X_train and X_test, plus the scaler object
X_train_model, X_test_model, scaler = scale_data(X_train, X_test, do_scaling)


# TABS

# Creates tabs for different sections 
tab1, tab2, tab3 = st.tabs(["Explore Data 🔍", "Model Guide 📖", "Results 📊"])

# TAB 1
# This section helps users understand their dataset before modeling

with tab1:
# This shows the title for this tab
    st.header("**Explores Your Data**")
# Shows some of the rows of the dataset
    st.dataframe(filtered_df.head(), hide_index=True)
# This shows some facts about the dataset, to make the user understand why the app made certain recommendations and to understand the dataset better
    metrics = {"Rows": filtered_df.shape[0],
        "Columns": filtered_df.shape[1],
        "Missing Values": filtered_df.isna().sum().sum(),
        "Unique Target Values": filtered_df[target].nunique(dropna=True)}
# This shows the metrics in columns
    cols = st.columns(len(metrics))
# This loops through to show every metric by matching the label and value with a column
# zip() puts the lists together 
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)

# Shows the data types of everyy column
    st.subheader("Column Types")
    st.write(filtered_df.dtypes)
# Shows the missing values for every column (adds them up)
    st.subheader("Missing Values by Column")
    missing_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Missing Values": filtered_df.isna().sum()
    }).sort_values("Missing Values")
    st.dataframe(missing_df)

# Will give visual of hwo the targer variable is distributed
    st.subheader("Target Distribution")
# fig is the container, ax is the bars, plt.sublot creates a new plot
    fig, ax = plt.subplots()
# If it is a classification model it will show a bar chart 
    if problem_type == "Classification":
# Counts the total observations in each class 
        counts = filtered_df[target].value_counts(dropna=False)
# Creates a bar chart
        ax.bar(counts.index.astype(str), counts.values)
        xlabel, ylabel, title = "Class", "Count", "Class Distribution"
        ax.tick_params(axis="x", rotation=45)
# If it is a regression then it will show a histogram of values
    else:
# Create a histogram with 20 bins
        ax.hist(y, bins=20)
        xlabel, ylabel, title = "Target Value", "Frequency", "Target Distribution"
# Set labels and show the plot 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)

# The heatmap will show the relationship between the variables
    st.subheader("Correlation Heatmap")

# Only numeric colums will be included
    corr = filtered_df.select_dtypes(include="number").corr()
# fig is the container, ax is the bars, plt.sublot creates a new plot
    fig, ax = plt.subplots()
    ax.imshow(corr)

# Set positions on the axes so column labels line up correctly 
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
# Label the axes with the column names
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

# Show the plot
    st.pyplot(fig)

# TAB 2
# Help users understand what type of machine learning problem they have

with tab2:
    st.header("Model Guide")

# Problem type first (classification or regression)
    st.subheader("Problem Type")
# Give definition of classification and regression (used Grok reading and Prof. Smiley's slides )
    st.write("**Classification** predicts categories or labels, such as spam vs. not spam, pass vs. fail, or yes vs. no.")
    st.write("**Regression** predicts continuous numeric values, such as house price, salary, or temperature." )
# Show the detected problem type based on the users dataset
    st.write(
        f"**Your detected problem type:** {problem_type} — your target variable is being "
        f"treated as {'categorical' if problem_type == 'Classification' else 'numeric'}." )


# Recommend model based on problem type
    st.subheader("Recommended Models")
# Loop through each recommendation and display model name and reason
    recommendations = recommend_models(problem_type, y)
    for name, reason in recommendations:
        st.write(f"**{name}:** {reason}")

# Model definitions provided for users to understand what each model does 
    st.subheader("Model Definitions")

    model_definitions = {
    "Logistic Regression": "A classification model that predicts the probability of a category.",
    "Decision Tree": "A model that makes predictions by splitting the data based on yes/no questions.",
    "KNN": "A model that predicts based on the closest data points.",
    "Linear Regression": "A regression model that predicts numeric values using a linear relationship."}
# Loops through each model and shows its name and definition
    for name, definition in model_definitions.items():
        st.write(f"**{name}:** {definition}")
    
# TAB 3

with tab3:
# Shows title for results
    st.header("Results")

# Give a disclaimer
    st.info("""
    DISCLAIMER: Not every model works with every dataset.

    For example:
    - Linear regression will not work if your target is not numeric
    - Classification needs at least two categories
    - Some models struggle if your data is not scaled
    - Missing values can break models

    If something does not run, it is usually because of one of these — try adjusting your settings!
    """)

# Does not show anyuthing if there is not a model that is selected
    if not model_choices:
        st.warning("Please select at least one model.")
        st.stop()

# This stores the results to compare the different models 
    results = []
# This stores the ROC curve 
    roc_results = []

# LOGISTIC REGRESSION

# Run logistic regression only if the user selected it
    if "Logistic Regression" in model_choices and problem_type == "Classification":
        st.subheader("Logistic Regression")
# Create and train the logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_model, y_train)
# Use the trained model to make predictions on the test data
        y_pred = model.predict(X_test_model)
# Calculate evaluation metrics to measure model performance
# zero_division=0 ensures stable metric calculation by returning 0 when precision or recall would be undefined
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
# Calculate evaluation metrics to measure model performance
        results.append({
            "Model": "Logistic Regression",
            "Accuracy": round(acc, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3)})
# Display the metrics 
        metric_values = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1}
# Show metrics next to each other
# zip() pairs pairs each column with a metric
        for col, (label, value) in zip(st.columns(4), metric_values.items()):
            col.metric(label, f"{value:.3f}")

# Confusion matrix to compare actual and predicted classes to show where the model messes up
        cm = confusion_matrix(y_test, y_pred)
# Creates and shows the confusion matrix on a plot
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

# Makes classification report
        st.text(classification_report(y_test, y_pred, zero_division=0))

# For binary classification calculatre ROC curve and AUC
        if y.nunique() == 2:
# Get predicted probabilites for the true class
            y_probs = model.predict_proba(X_test_model)[:, 1]
# Calculate false positive rate and trye positive rate
            fpr, tpr, _ = roc_curve(y_test, y_probs)
# Calculate area under the curve to know how well the model performed
            roc_auc = auc(fpr, tpr)

# Save ROC results so they can be plotted later
            roc_results.append({"name": "Logistic Regression", "fpr": fpr, "tpr": tpr, "auc": roc_auc})
            st.write(f"AUC: {roc_auc:.3f}")

# Show which features had the strongest influence on predictions
# Sort by the influence 
        coef_df = pd.DataFrame({
            "Feature": X_train_model.columns,
            "Coefficient": model.coef_[0]}).sort_values("Coefficient", key=np.abs, ascending=False)

        st.write("Top coefficients")
        st.dataframe(coef_df.head(10))

# DECISION TREE

# I forgot to add this earlier! (I didn't add the plot_tree)
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    
# Runs decision tree only if selected and problem type is classification
    if "Decision Tree (Classifier)" in model_choices and problem_type == "Classification":
        st.subheader("Decision Tree")
# Tunes hyperparemters using GridSearchCV if a person selects it
        if run_grid_search:
 # Try different values of max_depth and use cross-validation to find the best one
            grid = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
 # None = no limit, others control tree complexity
 # 5 times cross validation (cv=5)
                {"max_depth": [None, 3, 5]},cv=5)
# Fit grid search on training data
            grid.fit(X_train_model, y_train)
 # Use the best model found
            model = grid.best_estimator_
            st.write("Best parameters:", grid.best_params_)
# If not tuning, use the selected max_depth directly
        else:
            model = DecisionTreeClassifier(max_depth=max_depth)

# Train the model
        model.fit(X_train_model, y_train)
# Make predictions on the test data
        y_pred = model.predict(X_test_model)
# Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
# Store the results for comparing to other models
        results.append({"Model": "Decision Tree", "Accuracy": round(acc, 3)})
# Show the accuracy
        st.metric("Accuracy", f"{acc:.3f}")

# The confusion matrix shows the correct and incorrect predictions
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

        st.text(classification_report(y_test, y_pred, zero_division=0))

# Show the actual decision tree
        st.subheader("Decision Tree Diagram")
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(
            model,
            feature_names=X_train_model.columns,
            class_names=[str(c) for c in model.classes_],
            filled=True,
            rounded=True,
            ax=ax)
        st.pyplot(fig)

# KNN

# Run KNN only if the user selected it and the problem is classification
    if "KNN (Classifier)" in model_choices and problem_type == "Classification":
        st.subheader("KNN")
# Create the KNN model
        model = KNeighborsClassifier(n_neighbors=k)
# Train the model with training data 
        model.fit(X_train_model, y_train)
# Make predictions on test data
        y_pred = model.predict(X_test_model)
# Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": "KNN", "Accuracy": round(acc, 3)})
# Show accuracy on app
        st.metric("Accuracy", f"{acc:.3f}")

# K try different k values to see which performs best 
        k_vals = range(1, 15)
# Store accuracy scores for each k
        scores = []

# Loope through different k values
        for val in k_vals:
# Creat a model with those k's, train, predict, calculate, and store accuracy
            temp = KNeighborsClassifier(n_neighbors=val)
            temp.fit(X_train_model, y_train)
            pred = temp.predict(X_test_model)
            scores.append(accuracy_score(y_test, pred))

# Show results on plot
        fig, ax = plt.subplots()
        ax.plot(k_vals, scores)
        st.pyplot(fig)


 # LINEAR REGRESSION

# Run linear regression if its selected and the problem is regression
    if "Linear Regression" in model_choices and problem_type == "Regression":
        st.subheader("Linear Regression")
# Create the model
        model = LinearRegression()
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)

# Mean squared error gives the difference between actual and predicted values
        mse = mean_squared_error(y_test, y_pred)
# R2 measures how well the model explained the variation (0-1)
# Higher R2 is better
        r2 = r2_score(y_test, y_pred)
# Store results to compare with other models
        results.append({
            "Model": "Linear Regression",
            "MSE": round(mse, 3),
            "R2": round(r2, 3)})
# Show metrics in app
        st.metric("MSE", f"{mse:.3f}")
        st.metric("R²", f"{r2:.3f}")
# Show plot with labels
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)


# COMPARISON TABLE

# Only show this section if at least one model has been run
    if results:
        st.subheader("Comparison Table")
# Convert the results list into a pandas dataframe and show in app
        st.dataframe(pd.DataFrame(results))
