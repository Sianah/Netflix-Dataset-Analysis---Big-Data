import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import random
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
# REMEMEBER: Clustering is a technique in machine learning that involves grouping similar data points together.
# It is commonly used for data analysis, pattern recognition, and image processing.

# # Get a list of all the files in the HTRC dataset
# files = glob.glob('/c:/Users/Sisi/Desktop/Netflix Dataset Analysis - Big Data/netflix.csv')
# # Shuffle the files
# random.shuffle(files)
# # Print the first 10 files
# for file in files[:10]:
#     with open(file, 'r') as f:
#         for line in f:
#             data = line.strip().split(',')
#             print(data)
# df_netflix = pd.read_csv(filepath_or_buffer="netflix.csv", encoding='latin1')
# # print(df_netflix)

# # Check for missing values in the dataset
# missing_data_summary = df_netflix.isnull().sum()

# # Display the missing data summary
# print(missing_data_summary)

# load the dataset
netflix = pd.read_csv('netflix.csv')

# Display the first few rows of the dataset
categorical_columns = ['genre', 'language']
numerical_columns = ['imdb_score', 'runtime']

# Creating transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Creating a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Fit and transform the data
x_processed = preprocessor.fit_transform(netflix)

# determine the optimal number of clusters
inertia = []
k_range = range(1, 21)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_processed)
    inertia.append(kmeans.inertia_)

# Plot the inertia
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()

# Fit the KMeans algorithm to the data
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(x_processed)

# Add the cluster labels to the dataset
netflix['Cluster'] = clusters


genre_count = netflix.groupby(
    ['Cluster', 'genre']).size().unstack(fill_value=0)
genre_count.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Genre distribution by cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# scatter plot of imdb_score and runtime
plt.figure(figsize=(10, 6))
sns.scatterplot(data=netflix, x='runtime', y='imdb_score',
                hue='Cluster', palette='viridis', style='Cluster')
plt.title('IMDB Score vs Runtime')
plt.xlabel('Runtime')  # in minutes
plt.ylabel('IMDB Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# TODO:
# USE ANOTHER CLUSTERING TECHNIQUE(check)
# REDUCE DIMENSIONALITY TO TWO DIMENSIONS(check)
# GENERATE AUTOMATIC REPORTS(check)
# PERFORM CORRELATION ANALYSIS TO KNOW HOW DIFFERENT MOVIE CHARACTERISTICS ARE RELATED(check)
# VALIDATE THE CLUSTERING RESULTS USING THE SILHOUETTE SCORE(check)
# ANALYZE PREMIERE DATES AND YEARS TO IDENTIFY TRENDS(check)

# Fit and transform the data using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(x_processed)

# Add the cluster labels to the dataset
netflix['Cluster_DBSCAN'] = clusters

# Plot the clusters using DBSCAN
plt.figure(figsize=(10, 6))
sns.scatterplot(data=netflix, x='runtime', y='imdb_score',
                hue='Cluster_DBSCAN', palette='viridis', style='Cluster_DBSCAN')
plt.title('IMDB Score vs Runtime (DBSCAN)')
plt.xlabel('Runtime')  # in minutes
plt.ylabel('IMDB Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Reduce dimensionality to two dimensions
svd = TruncatedSVD(n_components=2)
x_svd = svd.fit_transform(x_processed)  # Apply SVD on the processed data
netflix['SVD1'] = x_svd[:, 0]
netflix['SVD2'] = x_svd[:, 1]

# Plotting DBSCAN results using the SVD reduced dimensions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SVD1', y='SVD2', hue='Cluster_DBSCAN', data=netflix, palette='viridis', style='Cluster_DBSCAN')
plt.title('DBSCAN Clustering Results')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.legend(title='DBSCAN Cluster')
plt.grid(True)
plt.show()

# generate automatic reports
# Display the first few rows of the dataset
print(netflix.head())
# Generate automatic reports
report = pd.DataFrame({'Cluster': netflix['Cluster'], 'Genre': netflix['genre'], 'Language': netflix['language'], 'IMDB Score': netflix['imdb_score'], 'Runtime': netflix['runtime']})
report.to_csv('netflix_report.csv', index=False)
print('Automatic report generated and saved as netflix_report.csv')
# Perform correlation analysis
numeric_netflix = netflix.select_dtypes(include=[np.number])
correlation_matrix = numeric_netflix.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Validate the clustering results using the silhouette score
silhouette = silhouette_score(x_processed, clusters)
print(f'Silhouette Score: {silhouette}')

# Analyze premiere dates and years to identify trends
# Extract the year from the premiere column
netflix['Year'] = pd.to_datetime(netflix['premiere']).dt.year
# Plot the number of movies released each year
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=netflix, palette='viridis',hue='year',legend=False)
plt.title('Number of Movies Released Each Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot the number of movies released each month
plt.figure(figsize=(10, 6))
month = pd.to_datetime(netflix['premiere']).dt.month
sns.countplot(x=month, data=netflix, palette='viridis', hue=month,legend=False)
plt.title('Number of Movies Released Each Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Plot the number of movies released each day of the week
plt.figure(figsize=(10, 6))
day = pd.to_datetime(netflix['premiere']).dt.day_name()
sns.countplot(x=day, data=netflix, palette='viridis', hue=day,legend=False)
plt.title('Number of Movies Released Each Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

#plot number of movies released each language
plt.figure(figsize=(10, 6))
sns.countplot(x='language', data=netflix, palette='viridis',hue='language',legend=False)
plt.title('Number of Movies Released Each Language')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()