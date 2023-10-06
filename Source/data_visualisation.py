import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Load the files
credits_df = pd.read_csv('./Data/credits.csv')
titles_df = pd.read_csv('./Data/titles.csv')

# Check for missing values in both dataframes
missing_values_credits = credits_df.isnull().sum()
missing_values_titles = titles_df.isnull().sum()

# Handle missing values

# Filling missing values in credits data
credits_df['character'].fillna('Not Specified', inplace=True)

# Filling missing values in titles data
titles_df['description'].fillna('No description available', inplace=True)
titles_df['age_certification'].fillna('Not Specified', inplace=True)

titles_df.drop(columns=['seasons'], inplace=True)

# Convert string representation of lists to actual lists
titles_df['genres'] = titles_df['genres'].apply(ast.literal_eval)
titles_df['production_countries'] = titles_df['production_countries'].apply(ast.literal_eval)

merged_df = titles_df.merge(credits_df, on='id', how='left')

# Setting the style for plots
sns.set_style("whitegrid")

# Distribution of Release Years
plt.figure(figsize=(12, 6))
sns.histplot(titles_df['release_year'], kde=False, bins=30, color='skyblue')
plt.title('Distribution of Release Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# Flatten the list of genres and count occurrences
all_genres = [genre for sublist in titles_df['genres'].tolist() for genre in sublist]
genres_count = pd.Series(all_genres).value_counts()

# Plot the most common genres
plt.figure(figsize=(12, 6))
genres_count.head(15).plot(kind='bar', color='skyblue')
plt.title('Top 15 Most Common Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of IMDb and TMDB Scores
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# IMDb Scores Distribution
sns.histplot(titles_df['imdb_score'], kde=True, ax=ax[0], color='skyblue', bins=20)
ax[0].set_title('Distribution of IMDb Scores')
ax[0].set_xlabel('IMDb Score')
ax[0].set_ylabel('Number of Movies')

# TMDB Scores Distribution
sns.histplot(titles_df['tmdb_score'], kde=True, ax=ax[1], color='salmon', bins=20)
ax[1].set_title('Distribution of TMDB Scores')
ax[1].set_xlabel('TMDB Score')
ax[1].set_ylabel('Number of Movies')

plt.tight_layout()
plt.show()

# Count occurrences of actors
actors_count = merged_df[merged_df['role'] == 'ACTOR']['name'].value_counts()

# Plot the most frequent actors
plt.figure(figsize=(12, 6))
actors_count.head(15).plot(kind='bar', color='lightgreen')
plt.title('Top 15 Most Frequent Actors')
plt.xlabel('Actor')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Flatten the list of production countries and count occurrences
all_countries = [country for sublist in titles_df['production_countries'].tolist() for country in sublist]
countries_count = pd.Series(all_countries).value_counts()

# Plot the most frequent production countries
plt.figure(figsize=(12, 6))
countries_count.head(10).plot(kind='bar', color='lightcoral')
plt.title('Top 10 Production Countries')
plt.xlabel('Country')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

