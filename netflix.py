import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit page configuration
st.set_page_config(page_title="Task Dashboard", layout="wide")

# Define color palette
color_palette = sns.color_palette("autumn", as_cmap=True)




# Load data
@st.cache_data
def load_data():
    # Replace this with actual data loading logic
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

    # Check if any CSV files are found
    if not csv_files:
        print("No CSV files found in the repository.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found

    dataframes = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
        df['ProjectID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + " " + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)
    combined_df['week'] = pd.to_datetime(combined_df['started_at'], errors="coerce").dt.isocalendar().week
    combined_df['month'] = pd.to_datetime(combined_df['started_at'], errors="coerce").dt.month
    combined_df['year'] = pd.to_datetime(combined_df['started_at'], errors="coerce").dt.year

    # Additional preprocessing logic
    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + " " + combined_df['id'].astype(str)
    combined_df['task_wo_punct'] = combined_df['task'].apply(lambda x: ''.join([char for char in str(x) if char not in string.punctuation]))
    combined_df['task_wo_punct_split'] = combined_df['task_wo_punct'].apply(lambda x: re.split(r'\W+', str(x).lower()))

    stopword = nltk.corpus.stopwords.words('english')
    combined_df['task_wo_punct_split_wo_stopwords'] = combined_df['task_wo_punct_split'].apply(
        lambda x: [word for word in x if word not in stopword]
    )

    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x]
    )

    combined_df["Hours"] = combined_df["minutes"] / 60
    combined_df["week"] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.isocalendar().week
    combined_df["month"] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.month
    combined_df["year_month"] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.to_period("M")

    categories = {
            "technology": ["website", "sql", "backend", "repository", "ai", "coding", "file", "database", "application", "program", "flask", "html", "css", "javascript"],
            "actions": ["reviewed", "created", "tested", "fixed", "debugged", "implemented", "researched", "planned", "updated", "designed", "documented", "analyzed", "optimized", "added", "removed"],
            "design": ["logo", "design", "styling", "layout", "responsive", "theme", "navbar", "icon", "image", "photo", "redesigning", "wireframes"],
            "writing": ["blog", "guide", "documentation", "report", "note", "summary", "draft", "content", "copywriting"],
            "meetings": ["meeting", "call", "discussion", "session", "presentation", "team"],
            "business": ["grant", "funding", "startup", "loan", "entrepreneur", "business", "government"],
            "errors": ["bug", "error", "issue", "fixing", "debugging", "problem", "mistake"],
            "time": ["hour", "day", "week", "month", "year"],
            "miscellaneous": []  # For words that don't fit into other categories
        }
    
    def categorize_words(task_wo_punct_split_wo_stopwords_lemmatized, categories):
            categorized = {category: [] for category in categories.keys()}
            uncategorized = []  # To track words that don't match any category

            for word in task_wo_punct_split_wo_stopwords_lemmatized:
                found = False
                for category, keywords in categories.items():
                    if word in keywords:
                        categorized[category].append(word)
                        found = True
                        break
                if not found:
                    uncategorized.append(word)  # Add to uncategorized if no match

            categorized["miscellaneous"] = uncategorized  # Add uncategorized words to "miscellaneous"
            return categorized

    combined_df['Categorized'] = combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: categorize_words(x, categories))

    return combined_df

# Load the data
combined_df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())
date_filter = st.sidebar.date_input("Filter by Date", [])
search_term = st.sidebar.text_input("Search Task", "")
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=combined_df['Full_Name'].unique())

# Filter data
filtered_data = combined_df.copy()
if categories:
    filtered_data = filtered_data[filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
        lambda x: any(word in categories for word in x)
    )]
if len(date_filter) == 2:
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    filtered_data = filtered_data[
        (filtered_data["started_at"] >= start_date) &
        (filtered_data["started_at"] <= end_date)
    ]
if search_term:
    filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]
if full_name_filter:
    filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]

# Tabs for graphs
tab1, tab2, tab3 = st.tabs(["Overview", "Hours", "Entries"])

# Tab 1: Overview
# Tab 1: Overview
with tab1:
    st.header("Top 20 Most Common Words")
    all_words = [word for sublist in filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'] for word in sublist]
    word_counts = Counter(all_words).most_common(20)
    
    if word_counts:
        words, counts = zip(*word_counts)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(words), y=list(counts), palette="autumn", ax=ax)
        ax.set_title("Top 20 Most Common Words (Lemmatized)", fontsize=14)
        ax.set_xlabel("Words", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.info("No data available for the selected filters to display word frequency.")


# Tab 2: Hours by Time Period
with tab2:
    st.header("Hours by Time Period")
    time_option = st.selectbox("Select Time Period", options=["Week", "Month", "Year"], index=2)
    
    if time_option == "Week":
        time_col = "week"
    elif time_option == "Month":
        time_col = "month"
    else:
        time_col = "year"

    grouped_data = filtered_data.groupby([time_col, 'Full_Name'])['Hours'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=grouped_data, x=time_col, y="Hours", hue="Full_Name", palette="autumn", ax=ax, dodge=False)
    ax.set_title(f"Hours by {time_option}", fontsize=14)
    ax.set_xlabel(time_option, fontsize=12)
    ax.set_ylabel("Hours", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Tab 3: Entries Count
with tab3:
    st.header("Entries Count by Time Period")
    counts_time_option = st.selectbox("Select Time Period", options=["Week", "Month", "Year"], index=1)
    
    if counts_time_option == "Week":
        count_col = "week"
    elif counts_time_option == "Month":
        count_col = "year_month"
    else:
        count_col = "year"

    counts = filtered_data.groupby(count_col)['ProjectID-ID'].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind='bar', color='#FE6E00', edgecolor='black', ax=ax)
    ax.set_title(f"Unique Entries by {counts_time_option}", fontsize=14)
    ax.set_xlabel(counts_time_option, fontsize=12)
    ax.set_ylabel("Unique Entries", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
