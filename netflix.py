import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="\ud83c\udfa5",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Netflix data
@st.cache
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("\ud83c\udfa5 Netflix Dashboard")
st.sidebar.markdown("Explore Netflix movies and TV shows!")

# Select filters
selected_type = st.sidebar.selectbox("Select Type", options=["All"] + list(df["type"].unique()))
selected_country = st.sidebar.selectbox("Select Country", options=["All"] + list(df["country"].dropna().unique()))
selected_year = st.sidebar.slider("Select Release Year", int(df["release_year"].min()), int(df["release_year"].max()), (2015, 2020))

# Filter data
filtered_data = df.copy()
if selected_type != "All":
    filtered_data = filtered_data[filtered_data["type"] == selected_type]
if selected_country != "All":
    filtered_data = filtered_data[filtered_data["country"] == selected_country]
filtered_data = filtered_data[(filtered_data["release_year"] >= selected_year[0]) & (filtered_data["release_year"] <= selected_year[1])]

# Display filtered data
st.write("### Filtered Netflix Data")
st.write(filtered_data)

# Visualization 1: Release Year Distribution
st.write("### Release Year Distribution")
release_year_chart = alt.Chart(filtered_data).mark_bar().encode(
    x=alt.X("release_year:O", title="Release Year"),
    y=alt.Y("count():Q", title="Count"),
    color="type:N",
    tooltip=["release_year", "count()"]
).properties(width=700, height=400)
st.altair_chart(release_year_chart)

# Visualization 2: Top Directors
st.write("### Top 10 Directors")
top_directors = filtered_data["director"].value_counts().head(10)
top_directors_chart = px.bar(top_directors, x=top_directors.index, y=top_directors.values, title="Top 10 Directors")
st.plotly_chart(top_directors_chart)

# Visualization 3: Genre Distribution
st.write("### Genre Distribution")
df_exploded_genre = filtered_data.assign(listed_in=filtered_data["listed_in"].str.split(",")).explode("listed_in")
df_exploded_genre["listed_in"] = df_exploded_genre["listed_in"].str.strip()
genre_counts = df_exploded_genre["listed_in"].value_counts().head(10)
genre_chart = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values, title="Top Genres")
st.plotly_chart(genre_chart)

# Visualization 4: Country Distribution
st.write("### Country Distribution")
df_exploded_country = filtered_data.assign(country=filtered_data["country"].str.split(",")).explode("country")
df_exploded_country["country"] = df_exploded_country["country"].str.strip()
country_counts = df_exploded_country["country"].value_counts().head(10)
country_chart = px.bar(country_counts, x=country_counts.index, y=country_counts.values, title="Top Countries")
st.plotly_chart(country_chart)

# Visualization 5: Heatmap
st.write("### Heatmap of Release Years vs. Ratings")
if "rating" in df.columns:
    heatmap_data = filtered_data.pivot_table(index="release_year", columns="rating", aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt="d", cbar=True, ax=ax)
    st.pyplot(fig)

# Visualization 6: Donut Chart (Type Distribution)
st.write("### Distribution by Type")
type_counts = filtered_data["type"].value_counts()
donut_chart = px.pie(type_counts, values=type_counts.values, names=type_counts.index, hole=0.5, title="Type Distribution")
st.plotly_chart(donut_chart)

# Save cleaned data
if st.sidebar.button("Save Cleaned Data"):
    filtered_data.to_csv("netflix_titles_cleaned.csv", index=False)
    st.sidebar.success("Cleaned data saved!")
