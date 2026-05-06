# Set up my code by importing necessary libraries 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# This gives the whole app a title
st.title("🔎 Uncovering the Unseen: Unsupervised Learning in Practice")
# This is basically the subtitle for the whole app
st.write("""Transform raw data into insight by exploring latent patterns with PCA, k-means, and hierarchical clustering.""")

# BACKGROUND DATASET SETUP 

# This gets streamlit to store uploaded dataset so it doesn't have to keep reloading
# The app uses pandas to read the CSV file 
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# This makes a copy of the dataset so it doesn't change the original dataset
def clean_data_for_modeling(df):
    model_df = df.copy()
# It goes through each column to clean each one
    for col in model_df.columns:
# This is datetime columns, it converts them into integer timestamps so the models can use the data
        if pd.api.types.is_datetime64_any_dtype(model_df[col]):
            model_df[col] = model_df[col].astype("int64") // 10**9
# This converts True/false columns into 0 or 1
        elif pd.api.types.is_bool_dtype(model_df[col]):
            model_df[col] = model_df[col].astype(int)
# This converts text columns into numerics 
        elif model_df[col].dtype == "object" or isinstance(model_df[col].dtype, pd.CategoricalDtype):
# This tries to convert strings into dates; the coerce is errors so if its an invalid bvalue it becomes NaT
            converted_date = pd.to_datetime(model_df[col], errors="coerce")
# This calculates the percentage of valid dates, so if more than 80% then it'll treat these columns as dates
# If not more than 80% it will just convert them as regular texts
            if converted_date.notna().mean() > 0.8:
# Converts dates into numeric timestamps
                model_df[col] = converted_date.astype("int64") // 10**9
# This is for converting categorical data into numbers
            else:
                codes, uniques = pd.factorize(model_df[col])
# If something is -1 it is missing, so it is converted into NaN so this value doesn't interfere with the models
                codes = pd.Series(codes).replace(-1, np.nan)
# This replaces original columns with the new numerical codes  
                model_df[col] = codes

# Drops anything non-numeric 
    model_df = model_df.select_dtypes(include=np.number)
# Returns cleaned dataset
    return model_df
# Converts dtype names into cleaner labels 
def simplify_dtype(dtype):
    if "int" in dtype:
        return "Integer"
    elif "float" in dtype:
        return "Float"
    elif "object" in dtype:
        return "Text"
    elif "bool" in dtype:
        return "Boolean"
    elif "datetime" in dtype:
        return "Datetime"
    elif "category" in dtype:
        return "Category"
    else:
        return "Other"
# Makes an empty list to store results
def create_type_summary(original_df, cleaned_df):
    summary = []
# Loops through columns
    for col in cleaned_df.columns:
# Get the original and cleaned types
        original_dtype_raw = str(original_df[col].dtype)
        cleaned_dtype_raw = str(cleaned_df[col].dtype)
# Simpligys these dataframes 
        original_dtype = simplify_dtype(original_dtype_raw)
        cleaned_dtype = simplify_dtype(cleaned_dtype_raw)
# This will show the user if a column needed to be transformed
        if original_dtype == cleaned_dtype:
            transformation = "No change"
        else:
            transformation = "Converted"
# This stores the results in a dictionary for every column
        summary.append({
            "Feature": col,
            "Original Data Type": original_dtype,
            "Cleaned Data Type": cleaned_dtype,
            "Transformation": transformation
        })
# this shows the list in a table
    return pd.DataFrame(summary)

# this is to scale the data with scikit-learn because we need to standardize the features
# It is converted back into a dataframe 
def scale_data(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return pd.DataFrame(scaled, columns=data.columns, index=data.index)

# UPLOADING THE DATA
# this makes the btton in the Streamlit app to upload a CSV file or chose 

st.subheader("Choose a Dataset")

dataset_choice = st.radio(
    "Use your own CSV file or try one of the sample datasets:",
    [
        "Upload my own CSV",
        "Sample Dataset 1: Titanic Dataset",
        "Sample Dataset 2: Penguins Dataset",
        "Sample Dataset 3: Iris Dataset"])

uploaded_file = None

if dataset_choice == "Upload my own CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = None
# calling the sample datasets so users can select them from my folder
elif dataset_choice == "Sample Dataset 1: Titanic Dataset":
    df = load_data("titanic-1.csv")

elif dataset_choice == "Sample Dataset 2: Penguins Dataset":
    df = load_data("penguins.csv")

elif dataset_choice == "Sample Dataset 3: Iris Dataset":
    df = load_data("Iris.csv")
# It won't run if there isn't a file uploaded
if df is not None:
# Takes the raw dataset and makes everything numeric
    numeric_df = clean_data_for_modeling(df)
# This makes sure there are at least 2 numeric columns
    if numeric_df.shape[1] < 2:
# This will show an error message and not run if there is not at least two features
# The st.stop prevents a crash
        st.error("Your dataset must include at least two numeric columns for this app to work.")
        st.stop()
# This creates the five tabs for the app 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 A Look at the Data",
        "📕 PCA",
        "📗 K-Means Clustering",
        "📘 Hierarchical Clustering",
        "📚 Model Summary"])


# TAB 1
# Have to indent everything underneath tab1 for it to appear in that tab
    with tab1:
        st.header("📖 A Look at the Data")
# This will explain a bit 
        st.write("""Review your dataset here before running PCA or clustering.
        See its size, missing values, cleaned features, and overall structure.""")
# This shows some rows of the original dataset so the user can see it before cleaning
        st.subheader("Preview the Raw Dataset")
        st.dataframe(df.head(), hide_index=True)

        st.subheader("Overview of Dataset")
# Makes a dictionary with the number of rows, columns, the total missing values, and the cleaned numeric columns that the app can use 
        metrics = {
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing Values": df.isna().sum().sum(),
            "Features Used for Modeling": numeric_df.shape[1]}
# make a column in for each metric in Streamlit
        cols = st.columns(len(metrics))
# Loops through every metric and shows it in streamlit
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, value)
# Makes a table that shows the missing values for every column and calculates the percent missing for each column
        st.subheader("Missing Values by Column")
        missing_table = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isna().sum().values,
            "Percent Missing": (df.isna().mean().values * 100).round(2)})
# Shows the missing values table 
        st.dataframe(missing_table, hide_index=True)
# Creates a chart 
        fig, ax = plt.subplots(figsize=(12, max(5, len(missing_table) * 0.4)))
# Makes this a horizontal bar chart 
        ax.barh(
            missing_table["Column"],
            missing_table["Missing Values"])
# Makes the labels for the chart 
        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Column")
        ax.set_title("Missing Values by Column")
# flips the y axis 
        ax.invert_yaxis()
# Loops through every bar in the chart
        for row, missing_value in enumerate(missing_table["Missing Values"]):
# Writes numnber at the end of the every bar
            ax.text(
                missing_value,
                row,
                str(missing_value),
# Center the label vertically 
                va="center")
# This is for fixing the spacing so the labels don't overlap 
        plt.tight_layout()
# Uses matplotlib for chart 
        st.pyplot(fig)
# This checks if there is no missing values in the raw dataset
        if df.isna().sum().sum() == 0:
            st.success("No missing values were found in this dataset.")
# If there is missing values, there will be a message
        else:
            st.info(
                "PCA and clustering need complete numeric data. Rows with missing values are removed so the results stay accurate.")

# New section
        st.subheader("Cleaned Features Used for Modeling")
# This makes the final dataset for modeling; the dropna() removes the rows that still have missing values and uses the cleaned numeric columns 
        model_data = numeric_df.dropna()
# Shows message with the total number of feautres with the complete rows left after taking out missing values 
        st.write(
            f"This app automatically converts usable non-numeric data into numeric data and uses all cleaned features: " 
            f"**{model_data.shape[1]} features** and **{model_data.shape[0]} complete rows**.")
# This shows the user how the data was transformed
        type_summary = create_type_summary(df, numeric_df)
        st.subheader("How Features Were Converted")
        st.dataframe(type_summary, hide_index=True)
# This shows the cleaned data that will be used for PCA, hiereachical, and k
        st.subheader("Preview of Cleaned Data for Modeling")
        st.dataframe(model_data.head(), hide_index=True)
# This makes sure that there is not less than two cleaned features because we need at least 2 for our models
        if model_data.shape[1] < 2:
            st.warning("Add at least two cleaned features to use PCA, k-means clustering, or hierarchical clustering.")
# This checks if there are less than three complete rows and shows a warning sign instead of the app crashing
        elif model_data.shape[0] < 3:
            st.warning("There needs to be at least three complete rows after removing missing values.")
# If there is at least two features and three complete rows, then it shows stats 
        else:
            st.subheader("Summary Statistics for Cleaned Features")
# Makes summary statistics
# .T flips the table for each feature to become its own row
# reset_index makes the feature names into a column
            basic_stats = model_data.describe().T.reset_index()
            basic_stats = basic_stats.rename(columns={
                "index": "Feature",
                "count": "Count",
                "mean": "Mean",
                "std": "Standard Deviation",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Maximum"})
# Shows the statistics table rounded to the second decimal place
            st.dataframe(basic_stats.round(2), hide_index=True)

            st.subheader("Interactive Scatterplot of Cleaned Data")
            st.write("Choose any two features to compare. This scatterplot shows how they relate using the cleaned data, with each point representing a single observation.")
# Makes two columns to select x-axis and y-axis
            col1, col2 = st.columns(2)
# Lets user pick a cleaned feature for x-axis
            with col1:
                x_axis = st.selectbox(
                    "Choose x-axis",
                    model_data.columns)
# Lets user pick a cleaned feature for y-axis
            with col2:
                y_axis = st.selectbox(
                    "Choose y-axis",
                    model_data.columns,
# index makes the y-axis the second column
                    index=1)

# makes a matplotlib plot with labels with the data the models will use 
            fig, ax = plt.subplots()
            ax.scatter(model_data[x_axis], model_data[y_axis], alpha=0.7)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Relationship Between {x_axis} and {y_axis}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# TAB 2 PCA
    with tab2:
# Title of tab
        st.header("📕 Principal Component Analysis")
# Explains what PCA is 
        st.write("""
        Principal Component Analysis (PCA) helps simplify large datasets by finding new directions in the data that capture the strongest overall patterns.
        Instead of treating every feature separately, PCA combines information from all the features into a smaller number of principal components.
               
        One way to think about PCA is that it “rotates” the dataset so the new axes line up with where the data varies the most. The first principal component captures the greatest amount of variation in the dataset. The second captures the next greatest amount while still measuring something different from the first, and this process continues for additional components.
        
        A major advantage of PCA is that it reduces overlap between features because the principal components are uncorrelated. This can make complicated datasets easier to visualize, easier to interpret, and more useful for clustering or other machine learning methods.
      
        This section uses all cleaned features from the dataset. The data is standardized beforehand so features with larger numeric scales do not overpower the analysis simply because their values are bigger.""")
        pca_data = numeric_df.dropna()
# Double checks that the dataset doesn't have less then two columns
        if pca_data.shape[1] < 2:
            st.warning("PCA requires at least 2 cleaned features.")
# Checks that the dataset doesn't have less then two rows because PCA needs at least that 
        elif pca_data.shape[0] < 3:
            st.warning("You need at least 3 complete rows to run PCA.")
# If the data has enough features and rows it will run 
        else:
# Standardizes the data to make the features comparable 
            pca_input = scale_data(pca_data)
# Stores the column names from cleaned dataset 
            feature_names = pca_data.columns
# finds maximum PCA components the data can have; it can't create more components than the number of feautures 
            max_components = min(len(feature_names), pca_input.shape[0])
# Makes PCA model that will reduce the data to two components 
            pca_2 = PCA(n_components=2)
# Fits and transforms the data
            pca_result_2 = pca_2.fit_transform(pca_input)
            pca_df = pd.DataFrame(
                pca_result_2,
                columns=["PC1", "PC2"])
# Makes a scatterplot using PC1 and PC2
            st.subheader("PCA: 2D Projection of the Data")
# Set up the plot
            fig, ax = plt.subplots()
            ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("PCA: 2D Projection")
            ax.grid(True)
            st.pyplot(fig)
# Describe what the scatterplot shows
            st.info("""
            This scatterplot shows the dataset projected onto the first two principal components, which are latent variables created by PCA from the original features.

            Each point represents one observation, while PC1 and PC2 capture the largest independent sources of variation in the data. 
            Reducing the dataset to these two latent variables helps reveal patterns, clusters, separation between groups, and potential outliers that may be difficult to detect in the original high-dimensional data.
            """)
# Explains what PC1 and PC2 are made up of
            st.subheader(" Feature Contributions to Principal Components (Loading Values)")
# Shows how each feature contributes to each principle component
            loadings_2 = pd.DataFrame(
                pca_2.components_.T,
                columns=["PC1", "PC2"],
                index=feature_names)
# Shows the values rounding to the third decimal
            st.dataframe(loadings_2.round(3))
# Share what the table is showing 
            st.info("""
            The table shows the loading values for PC1 and PC2. 
            A loading measures how strongly an original feature contributes to a principal component.

            Features with larger absolute loading values have greater influence on the component, 
            while values near zero contribute very little. Positive and negative values indicate direction: 
            features with the same sign move together along the component, while opposite signs indicate inverse relationships.
            """)
# This makes a chart, the more features the taller the chart is 
            fig_load, ax_load = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
# This makes y-positions for each feauture and the bar height controls the thickness of the bars
            y_pos = np.arange(len(feature_names))
            bar_height = 0.35
# This makes horizontal bars for PC1
            ax_load.barh(
                y_pos + bar_height / 2,
                loadings_2["PC1"],
                height=bar_height,
                label="PC1" )
# This makes horizontal bars for PC2 
# The plus/minus sign make the two bars show up next to each other for every feature 
            ax_load.barh(
                y_pos - bar_height / 2,
                loadings_2["PC2"],
                height=bar_height,
                label="PC2")
# Labels rows with feautre names
            ax_load.set_yticks(y_pos)
            ax_load.set_yticklabels(feature_names)
            ax_load.set_xlabel("Loading Value")
            ax_load.set_title("PCA Loadings")
# This adds a vertical line at zero so that its easier to see positive and negative 
            ax_load.axvline(0, linewidth=0.8)
            ax_load.legend()
            ax_load.invert_yaxis()
            st.pyplot(fig_load)
# Explain what the chart shows
            st.info("""
            The bar chart compares how strongly each feature contributes to PC1 and PC2. 
            Longer bars represent stronger influence on the component, making it easier to identify which variables are driving the main patterns captured by PCA.
            """)
# This makes the section where users can choose how many PCA components they want 
            st.subheader("Select the Number of Components")
# Checks whether they can have more than two components
            if max_components > 2:
# This makes the slider (can choose between 2-maximum possible components for a particular dataset)
                n_components = st.slider(
                    "Total PCA components",
                    min_value=2,
                    max_value=max_components,
                    value=2)
# If there are only 2 components they doesn't need to be a slider, the app will just use 2
            else:
                n_components = 2
# Runs PCA with number of components chosen by the user 
            pca_full = PCA(n_components=n_components)
            pca_full.fit(pca_input)
# calculates variation explained by each component, turned into a percentage 
            explained = pca_full.explained_variance_ratio_ * 100
# Calculates the running total 
            cumulative = np.cumsum(explained)
# Makes table showing every principal compoent, variance explained, cumulative variance)
            variance_table = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(n_components)],
                "Variance Explained (%)": explained.round(2),
                "Cumulative Variance (%)": cumulative.round(2)})
# Shows table 
            st.subheader("Variance Explained by Each Component")
            st.dataframe(variance_table, hide_index=True)
# Shows total variance explained by the total number of components
# cumulative[-1] takes the last item in indexing to get the total variance explained of all the components 
            st.metric(
                "Total Variance Explained",
                f"{cumulative[-1]:.2f}%")
    
# Elbow Chart
            st.subheader("Elbow Chart: PCA Variance Explained")
# Making PCA model, the components the user selects are the mass components 
            pca_elbow = PCA(n_components=max_components)
# The fit is learning the variance of the principle components with the standardized dataset 
            pca_elbow.fit(pca_input)
# Calculates total variance 
            cumulative_full = np.cumsum(pca_elbow.explained_variance_ratio_) * 100
# Creates the elbow chart with labels
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                range(1, len(cumulative_full) + 1),
                cumulative_full,
                marker="o")
            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Cumulative Explained Variance (%)")
            ax.set_title("PCA Variance Explained")
            ax.set_xticks(range(1, len(cumulative_full) + 1))
            ax.grid(True)
# Shows the chart 
            st.pyplot(fig)
# Explains what the elbow chart shows/how to interpret it
            st.info("""
            This chart shows the cumulative percentage of variance explained as more principal components are added. 
            The x-axis represents the number of components included, and the y-axis shows how much of the dataset’s total variation those components explain together.

            Look for the point where the curve starts to level off, often called the “elbow.” 
            Before this point, each additional component explains a meaningful amount of new information. 
            After the elbow, new components usually add only small improvements and may capture more noise than useful structure.
            """)

            st.subheader("Variance Explained by Each Component")
# For individual variance for each component
            explained_full = pca_elbow.explained_variance_ratio_ * 100
# Adds the total variance up 
            cumulative_full = np.cumsum(explained_full)
# Creates a sequence  from the components I have from the list of variance for each PCA component (Added 1 because range function doesn't include the stopping numnber)
            components = range(1, len(explained_full) + 1)

# Makes the chart 
            fig, ax1 = plt.subplots(figsize=(8, 6))
# The bars shows how much one component explains by itself 
            ax1.bar(
                components,
                explained_full,
                alpha=0.8,
                label="Individual Variance")
# Labels the axes
            ax1.set_xlabel("Principal Component")
            ax1.set_ylabel("Individual Variance Explained (%)")
            ax1.set_xticks(components)
            ax1.set_xticklabels([f"PC{i}" for i in components])
# Loops through each bar to add a percentage label 
            for i, v in enumerate(explained_full):
                ax1.text(
                    components[i],
                    v + 1,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9)
# Makes a y-axis for the cumulative variance to shhow on the same chart
            ax2 = ax1.twinx()
# Plots cumulative variance line
            ax2.plot(
                components,
                cumulative_full,
                marker="o",
                label="Cumulative Variance")
# Labels second y-axis for the cumaltive variance line
            ax2.set_ylabel("Cumulative Variance Explained (%)")
# Keeps cumulative variance scale between 0-100 percent
            ax2.set_ylim(0, 100)
# Gets legend info from the bar chart and cumulative variance line
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
# Combines the legends to differentiate the bars from the line 
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="center right",
                bbox_to_anchor=(0.85, 0.5))
# Adds a title to the full chart
            plt.title("PCA: Variance Explained", pad=20)
# Fixes spacing and overlap
            plt.tight_layout()
# Shows full chart
            st.pyplot(fig)
# Explain what the chart is showing
            st.info("""
            This chart combines two views of PCA variance. 
            The bars show how much variation each principal component explains by itself, while the line shows the total variation explained as the components are added together.

            Using both views makes it easier to see which components carry the most information and when later components start adding very little. 
            Strong early bars mean those components explain major patterns in the dataset, while a flattening cumulative line shows that additional components are contributing less new information.
            """)

# TAB 3: K-MEANS CLUSTERING
    with tab3:
# All indented into the kmeans tab
        st.header("📗 K-Means Clustering")
        st.write("""
        K-Means groups observations into a chosen number of clusters based on how similar their feature values are. 
        It starts by placing cluster centroids, measures the distance from each observation to those centroids, and assigns each point to the nearest cluster.

        After the assignments are made, the algorithm recalculates each centroid using the mean position of the points inside that cluster. 
        This process repeats until the centroids stop moving noticeably and the cluster assignments stabilize.

        The overall goal is to create clusters where observations inside the same group are as similar as possible, while keeping different clusters as separated as possible.
        
        This section uses all cleaned features from the dataset. The data is standardized beforehand so features with larger numeric scales do not overpower the analysis simply because their values are bigger.""")
# Uses numeric columns and drops rows with missing values so the model doesn't break
        kmeans_data = numeric_df.dropna()
# k-means needs at least two features to form clusters , if there aren't features it will show a warning
        if kmeans_data.shape[1] < 2:
            st.warning("K-means requires at least 2 cleaned features.")
# Standardize data foro accurate clustering
        else:
            kmeans_input = scale_data(kmeans_data)
# this prevents to many clusters 
            max_k = min(10, len(kmeans_input) - 1)
# Need at least three rows to test multiple cluster options
            if max_k < 2:
                st.warning("You need at least 3 complete rows to run k-means.")
# Random state is set to 42 
            else:
                random_state = 42

                st.subheader("Choose the Best Number of Clusters")
                st.write("""
                To pick a good value for k, don’t rely on just one metric—look at both the elbow plot and the silhouette scores together.
                
                The elbow plot shows when adding more clusters stops making a big difference. The silhouette score shows how cleanly the data is grouped.
                        
                A good choice is usually where the elbow starts to flatten and the silhouette score is still relatively high.""")
# Builds sequence with clusters  to test for the best k 
                k_values = range(2, max_k + 1)
# Create empty lists to store results for WCSS and silhouette 
                inertia_values = []
                silhouette_values = []
# For every possible number of clusters make a k-means model to test how good it is 
# n_init is set at 20 so that the k-means is more reliable because it runs multiple times with different starting points and choses the best clustering
                for current_k in k_values:
                    temp_model = KMeans(
                        n_clusters=current_k,
                        random_state=random_state,
                        n_init=20)
                    temp_labels = temp_model.fit_predict(kmeans_input)
# Saves the value for each k of how close together the clusters are (want a lower value)
                    inertia_values.append(temp_model.inertia_)
# Evaluates how each point fits in its cluster (how seperated clusters are from each other)
                    silhouette_values.append(
                        silhouette_score(kmeans_input, temp_labels))
# This combines everything 
                evaluation_df = pd.DataFrame({
                    "Number of Clusters (k)": list(k_values),
                    "Within-Cluster Sum of Squares (WCSS)": inertia_values,
                    "Silhouette Score": silhouette_values})
# Shows results to the user 
                st.dataframe(evaluation_df.round(3), hide_index=True)
                st.subheader("Elbow Plot")
# Sets the x-axis with the number of clusters and the WCSS as the y-acis
                fig, ax = plt.subplots()
                ax.plot(list(k_values), inertia_values, marker="o")
# Labels the axes and gives a title
                ax.set_xlabel("Number of Clusters (k)")
                ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
                ax.set_title("Elbow Method for Optimal k")
# Shows plot on streamlit
                st.pyplot(fig)
# Explain how to interpret the elbow plot
                st.info("""
                The elbow plot helps show when adding more clusters stops improving the model by a meaningful amount. 
                At first, adding clusters usually reduces within-cluster distance a lot, but after a certain point the improvements become much smaller. 
                That bend in the curve is the “elbow.”

                The silhouette score helps measure how clearly separated the clusters are. 
                Higher scores mean points are generally closer to their own cluster and farther from neighboring clusters.

                The best value for k is usually near the elbow, especially if the silhouette score is still relatively high at that point.
                """)
                st.subheader("Silhouette Score Plot")
                fig, ax = plt.subplots()
# Sets the x-axis with k and y with the silhoutte score 
                ax.plot(list(k_values), silhouette_values, marker="o")
# Labels ths graph and gives title
                ax.set_xlabel("Number of Clusters (k)")
                ax.set_ylabel("Silhouette Score")
                ax.set_title("Silhouette Scores for Optimal k")
# Shows plot on streamlit
                st.pyplot(fig)
# idxmax() finds the row with the highest silhouette score 
# .loc get the the k value with that score, so it automatically picks the optimal k 
                best_silhouette_k = evaluation_df.loc[
                    evaluation_df["Silhouette Score"].idxmax(),
                    "Number of Clusters (k)"]
#Explains the elbow and silhoutette
                st.info(
                f"The silhouette score is highest at k = {best_silhouette_k}. "
                "The silhouette score measures how closely each point matches its own cluster compared to neighboring clusters. "
                "Higher values usually mean the clusters are more compact and better separated from each other. ")
# This makes the section where users can choose how many PCA components they want 
            st.subheader("Select the Number of Clusters")
# Checks whether they can have more than two components
            if max_k > 2:
# Interactive slider that defaults to the best silhoutte k, but the user can change it
                k = st.slider(
                    "Choose the number of clusters (k)",
                    min_value=2,
                    max_value=max_k,
                    value=int(best_silhouette_k))
# This buulds the final clustering model by assignign each data point to a cluster
# n_init is set at 20 to run k mean 20 times to find the k with the lowest inertia
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=20)

# Fits k-mean model to data and assigns every row to a cluster
                kmeans_labels = kmeans.fit_predict(kmeans_input)

                st.subheader("Size of Clusters")
# Converts labels into panda series, counts the points in every cluster
# Sort_index puts the clusters in order 
                cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
# This puts the data into a table
                cluster_sizes_df = cluster_sizes.reset_index()
                cluster_sizes_df.columns = ["Cluster", "Count"]
# Shows the table in streamlit
                st.dataframe(cluster_sizes_df, hide_index=True)

                st.subheader("K-Means Scatterplot Using PCA Projection")
# Compresses daya into PC1 and 2 
                pca_cluster = PCA(n_components=2)
                kmeans_pca_result = pca_cluster.fit_transform(kmeans_input)
# Makes a datafram with PC1 as the x-axis and PC2 as the y-axis
                kmeans_pca_df = pd.DataFrame(
                    kmeans_pca_result,
                    columns=["PC1", "PC2"],
                    index=kmeans_input.index)
# This labels each point by cluster
                kmeans_pca_df["Cluster"] = kmeans_labels
# Makes plot
                fig, ax = plt.subplots()
# Loops through each cluster
                for cluster_label in np.unique(kmeans_labels):
# Filters data 
                    cluster_data = kmeans_pca_df[kmeans_pca_df["Cluster"] == cluster_label]
# Details the plot with the borders, legend, position
                    ax.scatter(
                        cluster_data["PC1"],
                        cluster_data["PC2"],
                        alpha=0.7,
                        edgecolor="k",
                        s=60,
                        label=f"Cluster {cluster_label}")
# Labels the axes and title
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.set_title(f"K-means Clusters on 2D PCA Projection with k = {k}")
# Adds a legend
                ax.legend(loc="best")
# Adds a grid to make it easier to read 
                ax.grid(True)
# Shows the grid on the app
                st.pyplot(fig)
                st.info("""
                PCA is *only* for display; it was **not** used to fit the clusters.
                
                The colors show the K-Means cluster assignments based on the number of clusters selected by the user.""")

# TAB 4 HIERARCHICAL CLUSTERING
    with tab4:
        st.header("📘 Hierarchical Clustering")
# Explain hierarchical clustering
        st.write("""
        Hierarchical clustering builds clusters step by step by merging the most similar observations together. 
        Initially, each observation starts as its own cluster, and larger groups form as merges continue.

        This section uses **Ward linkage**, which is usually the strongest choice for numeric datasets because it minimizes increases in within-cluster variance. 
        In practice, this means it tries to keep clusters compact, balanced, and internally similar rather than simply connecting nearby points.

        Ward linkage is generally preferred over:
        - **single linkage**, which can create long “chains” of loosely connected points
        - **complete linkage**, which can become sensitive to outliers
        - **average linkage**, which balances distances but does not directly minimize cluster variance

        Because of this, Ward linkage often produces cleaner cluster boundaries and more interpretable dendrograms for structured numeric data.
        
        This section uses all cleaned features from the dataset. The data is standardized beforehand so features with larger numeric scales do not overpower the analysis simply because their values are bigger.""")
# Uses clean numeric dataset and remove missing rows with dropna()
        h_data = numeric_df.dropna()
# Checks that there are at least two features
        if h_data.shape[1] < 2:
            st.warning("You need at least 2 cleaned features for hierarchical clustering.")
# Checks if there are enough rows 
        elif len(h_data) < 3:
            st.warning("You need at least 3 rows after removing missing values.")
# Standardizes the data before clustering 
        else:
            h_input = scale_data(h_data)

            st.subheader("Dendrogram")
# Creates a copy of the scaled data for the dendrogram
            dendrogram_data = h_input.copy()
# Computes the linkage matrix
            Z = linkage(dendrogram_data, method="ward")
# Creates the plot
            fig, ax = plt.subplots(figsize=(20, 7))
# Plots the dendrogram with labels (uses rows)
            dendrogram(
                Z,
                ax=ax,
                labels=dendrogram_data.index.astype(str).tolist(),
                leaf_rotation=90, leaf_font_size=6)
            ax.set_title("Hierarchical Clustering Dendrogram Using Ward Linkage")
            ax.set_xlabel("Data Points")
            ax.set_ylabel("Distance")
# Shows the plot in the app
            st.pyplot(fig)
# Explains dendrogram
            st.info("""
            The dendrogram visualizes the full merge history created during agglomerative hierarchical clustering. 
            At the start, every observation is treated as its own cluster, and the algorithm repeatedly merges the most similar clusters together to build a nested tree structure.

            Each vertical merge represents a clustering step, and the height of the merge shows the linkage distance between the groups being combined. 
            Smaller merge heights indicate more similar observations or clusters, while large vertical jumps suggest the algorithm is merging less similar groups.

            Because the dendrogram preserves the full clustering history, it allows the number of clusters to be chosen after examining the structure of the data rather than fixing k in advance.""")
           
            st.subheader("Select the Number of Clusters")
# This prevents more clusters than data points from being an option
            max_h_k = min(10, len(h_input) - 1)
# Makes a slider to pick clusters (it just starts at 4, but users can change it)
            h_k = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=max_h_k,
                value=min(4, max_h_k))
# Makes hierachical model bottom up (agglomerative) 
            h_model = AgglomerativeClustering(
                n_clusters=h_k,
                linkage="ward")
# Fits the model to the data and returns a label for every row 
            h_labels = h_model.fit_predict(h_input)

            st.subheader("Cluster Size Count")
# Counts how many points are in each cluster, which is then put into a table with .reset_index
            h_cluster_sizes_df = (
                pd.Series(h_labels)
                .value_counts()
                .sort_index()
                .reset_index())
# Makes the table clearer
            h_cluster_sizes_df.columns = ["Cluster", "Count"]
# Shows the cluster size in app 
            st.dataframe(h_cluster_sizes_df, hide_index=True)

            st.subheader("Silhouette Score")
# This calculates the silhouette score 
            h_sil_score = silhouette_score(h_input, h_labels)
# Shows the score as a streamlit metric rounded to third decimal place
            st.metric("Silhouette Score with Number of Selected Clusters", f"{h_sil_score:.3f}")
# Explain what the score means 
            st.info("""Higher scores mean better-separated clusters.""")

            st.subheader("Silhouette Scores Across Cluster Counts")
# Makes clusters to test, starting at two because thats the minimum to have a functional silhouette score
            k_range = range(2, max_h_k + 1)
# Empty list to store the silhouette score for each cluster count
            sil_scores = []
# Loop through every number of clusters
            for k in k_range:
# For every k value this runs hierarchical clustering, the fit_preduct trains the model and assigns the rows into clusters
                temp_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(h_input)
# Calculates the silhouette score for cluster, then it adds it to the list 
                sil_scores.append(silhouette_score(h_input, temp_labels))
# Finds the highest silhoutte score for the total clusters 
            best_k = list(k_range)[np.argmax(sil_scores)]
# Creates the plot and makes sure the score changes as the number of clusters changes 
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(list(k_range), sil_scores, marker="o")
# Makes sure each tested total clusters shows up on the x-acis 
            ax.set_xticks(list(k_range))
# Label axes and title
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Analysis (Ward Clustering)")
# Light grid for the plot
            ax.grid(True, alpha=0.3)
# Show the plot in app
            st.pyplot(fig)
# Show the best cluster amount based on the highest silhouette
            st.success(
                f"Best number of clusters: {best_k} (score = {max(sil_scores):.3f})")


            st.subheader("Hierarchical Clusters on PCA Scatterplot")
# Makes PCA model
            pca = PCA(n_components=2)
# This makes the MCA model reducing the dataset into 2 components (PC1 and 2)
            h_pca = pca.fit_transform(h_input)
# Makes the plot
            fig, ax = plt.subplots(figsize=(8, 6))
# Loops through every cluster
            for cluster_label in np.unique(h_labels):
# Only picks the rows that belong to a particular cluster
                mask = h_labels == cluster_label
# The clusters get plotted seperately and gives label on legend based on color 
                ax.scatter(
                    h_pca[mask, 0],
                    h_pca[mask, 1],
                    alpha=0.7,
                    edgecolor="k",
                    s=60,
                    label=f"Cluster {cluster_label}")
# Labels the axes and the title 
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("Hierarchical Clustering (Ward Linkage) with PCA")
            ax.legend()
            ax.grid(True, alpha=0.3)
# shows the plot in streamlit
            st.pyplot(fig)
# Explain the scatterplot
            st.info("""
            PCA is *only* for display; it was **not** used to fit the clusters.
            
            Each point represents one observation, and point colors indicate the cluster assignments produced by hierarchical clustering.""")

# TAB 5: MODEL SUMMARY
    with tab5:
        st.header("📚 Model Summary")

        st.write("""
        This final section summarizes what each method is meant to do, what it gives back, 
        and what to watch out for when interpreting the results.""")

# I made little drop down sections for the three models to summarize them and share their limitations
        with st.expander("Principal Component Analysis"):
            st.markdown("""
            **Objective:**  
            PCA takes all cleaned numeric features and creates new summary variables called principal components. 
            These components are ordered by how much variation they explain, so PC1 captures the strongest overall pattern, PC2 captures the next strongest pattern, and so on.

            **What to look for:**  
            In the variance charts, check how quickly the cumulative variance increases. 
            If the first few components explain a large share of the variation, the dataset has a strong underlying structure. 
            If the variance is spread across many components, the patterns are more complex and harder to summarize.

            **Why it is useful:**  
            PCA makes high-dimensional data easier to inspect because it compresses many features into a smaller number of latent variables. 
            The PCA scatterplot can reveal separation, overlap, unusual observations, or possible cluster structure before running clustering methods.

            **Limitations:**  
            PCA does not create clusters by itself. It also does not tell you whether a pattern is meaningful in real-world terms. 
            The components are combinations of the original features, so they need to be interpreted through the loading values.

            **Results it gives:**  
            Explained variance, cumulative variance, 2D PCA projection, and loading values showing which original features contribute most to PC1 and PC2.
            """)
                
        with st.expander("K-Means Clustering"):
            st.markdown("""
            **Objective:**  
            K-Means partitions the dataset into a chosen number of clusters by assigning observations to the nearest centroid based on feature similarity. 
            The algorithm repeatedly updates cluster centers until the assignments stabilize.

            **What to look for:**  
            Use the elbow plot to identify where adding more clusters stops meaningfully reducing within-cluster variation. 
            Then compare the silhouette scores to see whether the clusters are actually well separated. 
            Strong clustering results usually show both a visible elbow and relatively high silhouette scores.

            **Why it is useful:**  
            K-Means works well when the dataset contains compact groups with relatively clear boundaries. 
            It creates direct cluster assignments, making it easier to identify groups of observations with similar feature patterns.

            **Limitations:**  
            K-Means requires the number of clusters (k) to be chosen in advance. 
            It can also struggle when clusters have irregular shapes, very different sizes, overlapping boundaries, or strong outliers because the algorithm relies heavily on distance from the centroids.

            **Results it gives:**  
            Cluster labels, inertia values, elbow plot, silhouette scores, centroid-based grouping, and PCA scatterplots colored by cluster assignment.
            """)

        with st.expander("Hierarchical Clustering"):
            st.markdown("""
            **Objective:**  
            Hierarchical clustering builds nested clusters by repeatedly merging the most similar observations or groups. 
            In this app, Ward linkage is used to minimize increases in within-cluster variance and keep clusters compact.

            **What to look for:**  
            Focus on the dendrogram structure and the height of the merges. 
            Large vertical jumps usually indicate the algorithm is combining less similar groups, which can help identify reasonable places to separate the data into clusters.

            **Why it is useful:**  
            Unlike K-Means, hierarchical clustering does not require a fixed number of clusters before the model begins. 
            It preserves the full merge history of the dataset, making it useful for exploring multi-level structure, subgroup relationships, and possible outliers.

            **Limitations:**  
            Hierarchical clustering can become computationally expensive on larger datasets, and dense dendrograms may become difficult to interpret when there are many observations. 
            Early merges also cannot be reversed later in the clustering process.

            **Results it gives:**  
            Dendrogram, hierarchical cluster assignments, silhouette scores, and PCA scatterplots colored by cluster membership.
            """)
        
        st.info("""
        A good approach is to use PCA first to get a sense of the main patterns and structure in the dataset, 
        then compare K-Means and hierarchical clustering to see whether similar groups appear across both methods. 
        If both approaches produce clear and consistent clusters, that usually gives more confidence that the grouping structure is meaningful.""")
# IF there is no dataset selected it won't run
else:
    st.info("Upload a CSV file or choose one of the sample datasets to begin.")
