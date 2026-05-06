import streamlit as st
import pandas as pd

# Load the CSV file
df = pd.read_csv("data/penguins.csv")


# this is how I made the sidebar
st.sidebar.title("Table of Contents")
# the radio buttons are to select a specific page
different_pages = st.sidebar.radio(
    "Select a page:", 
    ["Exploring Our Dataset", "Body Variation", "Key Bodily Correlations"])


# This is the section header for the filters
st.sidebar.markdown("# Filters")

# The boxes let you pick which penguins you want to look at
# dropna() makes sure we don’t include penguins with missing information
# default= makes all the choices selected at first so you see everything
filter_for_species = st.sidebar.multiselect("species", df["species"].dropna().unique(), default=df["species"].dropna().unique())
filter_for_island = st.sidebar.multiselect("island", df["island"].dropna().unique(), default=df["island"].dropna().unique())
filter_for_sex = st.sidebar.multiselect("sex", df["sex"].dropna().unique(), default=df["sex"].dropna().unique())

# Made a mask to filter the dataframe based on the selected characteristics
mask = (
    (df["species"].isin(filter_for_species)) &
    (df["island"].isin(filter_for_island)) &
    (df["sex"].isin(filter_for_sex))
)

# I apply the mask with this 
filtered_df1 = df[mask]

import matplotlib.pyplot as plt
# this set up a dark background, I thought it looked better aesthetically
plt.style.use("dark_background")

# This is Page 1
if different_pages == "Exploring Our Dataset":
    st.title("Penguins Insights Dashboard 🐧")
    st.write("This dashboard explores physical differences among Adelie, Chinstrap, and Gentoo penguins, focusing on variation in body mass and flipper length across species and between sexes.")

# Interactive button that shows a fun penguin fact
    if st.button("Fun penguin fact!"):
        st.write("Adelie and Gentoo suitors offer pebbles to potential mates. Acceptance signals the start of their bond and nest construction.")

# New name for metrics 
    metrics = {
        "Total Penguins": len(filtered_df1),
        "Species Count": filtered_df1["species"].nunique(),
        "Average Body Mass": round(filtered_df1["body_mass_g"].mean(), 1)
    }

    for col, (k, v) in zip(st.columns(3), metrics.items()):
        col.metric(k, v)
    
    species_counts = filtered_df1["species"].value_counts()

    # creates a new chart to add all of the labels
    fig, ax = plt.subplots()

    ax.bar(species_counts.index, species_counts.values)


    ax.set_xlabel("Species", labelpad=20)
    ax.set_ylabel("Number of Penguins")
    ax.set_title("Penguin Count by Species")

     # makes axis starts at 0 because before it was -20
    ax.set_ylim(bottom=0) 

    st.pyplot(fig)


# Page 2
import matplotlib.pyplot as plt

plt.style.use("dark_background")

if different_pages == "Body Variation": 
    st.title("Body Variation")
    st.markdown("Examine how body mass and flipper length differ between species and sexes.")

# showing the average body mass by species category 

    st.subheader("Average Body Mass by Species")

    average_mass = filtered_df1.groupby("species")["body_mass_g"].mean()
    
    fig, ax = plt.subplots()
    ax.bar(average_mass.index, average_mass.values)

    ax.set_xlabel("Species", labelpad=20) 
    ax.set_ylabel("Average Body Mass (g)")
    ax.set_title("Average Body Mass by Species")
    ax.set_ylim(bottom=0)

    st.pyplot(fig)
    
    st.caption("On average, the Gentoo penguins are heavier than the other two species.")
   
    st.divider()

# showing the average flipper length by species category 

    st.subheader("Average Flipper Length by Species")

    average_flipper = filtered_df1.groupby("species")["flipper_length_mm"].mean()
    
    fig, ax = plt.subplots()
    
    ax.bar(average_flipper.index, average_flipper.values)

    ax.set_xlabel("Species", labelpad=20)
    ax.set_ylabel("Average Flipper Length (mm)")
    ax.set_title("Average Flipper Length by Species")
    ax.set_ylim(bottom=0)

    st.pyplot(fig)

    st.caption("On average, the Gentoo Species has the longest flippers.")   

    st.divider()

# showing the average body mass difference between male and female penguins

    st.subheader("Male vs. Female Body Mass")

    gender_mass = filtered_df1.groupby("sex")["body_mass_g"].mean()
    
    fig, ax = plt.subplots()

    ax.bar(gender_mass.index, gender_mass.values)

    ax.set_xlabel("Sex")
    ax.set_ylabel("Average Body Mass (g)")
    ax.set_title("Average Body Mass by Sex")
    ax.set_ylim(bottom=0)

    st.pyplot(fig)

    st.caption("Based on the data, males typically have a higher body mass compared to female penguins.")

# This is for page 3
elif different_pages == "Key Bodily Correlations":
    st.title("Key Bodily Correlations")
    st.markdown("Explore how bill size, flipper length, and body mass relate across species and sexes.")

# I am renaming the datafram column names to be more professional on the app
    column_names= {
        "bill_length_mm": "Bill Length",
        "bill_depth_mm": "Bill Depth",
        "flipper_length_mm": "Flipper Length",
        "body_mass_g": "Body Mass"
    }



# I made a list of the penguin traits I want to look at 
    numeric_cols = list(column_names.keys())

# this put two boxes next to each other to pick the traits to compare 
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("X-axis", numeric_cols, index=2, format_func=lambda x: column_names[x])
    y_axis = col2.selectbox("Y-axis", numeric_cols, index=3, format_func=lambda x: column_names[x])

    st.subheader(f"{column_names[x_axis]} vs {column_names[y_axis]}")

# only keep the data we need and skip any incomplete rows so nothing breaks in the chart
    scatter_data = filtered_df1[[x_axis, y_axis, "species"]].dropna()

# made the scatter plot 
    st.scatter_chart(
        scatter_data,
        x=x_axis,
        y=y_axis,
        color="species"
    )
