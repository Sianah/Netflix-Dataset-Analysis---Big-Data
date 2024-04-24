import glob
import random
import bz2
import json
import numpy as np
import pandas as pd

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
df_netflix = pd.read_csv(filepath_or_buffer="netflix.csv", encoding='latin1')
print(df_netflix)
