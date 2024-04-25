import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
#REMEMEBER: Clustering is a technique in machine learning that involves grouping similar data points together. 
#It is commonly used for data analysis, pattern recognition, and image processing.

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

#load the dataset
netflix = pd.read_csv('netflix.csv')

# Display the first few rows of the dataset
categorical_columns = ['genre', 'language']
numerical_columns = ['imdb_score', 'runtime']