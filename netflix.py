import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
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


genre_count = netflix.groupby(['Cluster','genre']).size().unstack(fill_value=0)
genre_count.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Genre distribution by cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()