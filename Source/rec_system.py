import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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

#Dropping the 'seasons' column, because we are interested in movies
titles_df.drop(columns=['seasons'], inplace=True)

# Convert string representation of lists to actual lists
titles_df['genres'] = titles_df['genres'].apply(ast.literal_eval)
titles_df['production_countries'] = titles_df['production_countries'].apply(ast.literal_eval)

#merege 'titles' dataframe with 'credits' dataframe
merged_df = titles_df.merge(credits_df, on='id', how='left')


# Using TF-IDF to vectorize the movie descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['description'])

# Convert the matrix to a dataframe for easier manipulation later on
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# One-Hot Encoding for genres
genres_df = titles_df['genres'].apply(lambda x: '|'.join(x)).str.get_dummies(sep='|')

# Aggregating actor information for each movie
actor_aggregation = merged_df.groupby('id')['name'].apply(list).reset_index()
actor_aggregation.columns = ['id', 'actors']

# Aggregating role information for each movie
role_aggregation = merged_df.groupby('id')['role'].apply(list).reset_index()
role_aggregation.columns = ['id', 'roles']

# Merging the actor and role aggregations
cast_df = actor_aggregation.merge(role_aggregation, on='id')

# List of numerical columns to normalize
numerical_cols = ['imdb_score', 'tmdb_score', 'tmdb_popularity', 'runtime']

# Normalize the numerical columns
scaler = MinMaxScaler()
titles_df[numerical_cols] = scaler.fit_transform(titles_df[numerical_cols])

# Concatenate all the features into a single dataframe
features_df = pd.concat([tfidf_df, genres_df, titles_df[numerical_cols]], axis=1)

# Drop rows containing NaN values
cleaned_sample_features_df = features_df.dropna()

# Check the shape after dropping rows
cleaned_sample_shape = cleaned_sample_features_df.shape

# Recompute the cosine similarity matrix for the cleaned sample dataset
cleaned_sample_similarity_matrix = cosine_similarity(cleaned_sample_features_df)

# Convert the matrix to a dataframe for easier querying
cleaned_sample_similarity_df = pd.DataFrame(cleaned_sample_similarity_matrix, index=titles_df.loc[cleaned_sample_features_df.index]['title'], columns=titles_df.loc[cleaned_sample_features_df.index]['title'])

# Ask the user for movie title and number of recommendations
movie_title = input("Enter the title of the movie: ")
number_of_recommendations = int(input("Enter the number of recommendations you want: "))

# Get the movies with the highest similarity scores
recommended_movies = cleaned_sample_similarity_df[movie_title].sort_values(ascending=False)[1:number_of_recommendations+1]
print(recommended_movies)