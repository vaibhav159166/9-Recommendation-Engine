# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:16:11 2023

@author: Rahul Raje
"""

"""
Problem Statement: -

The Entertainment Company, which is an online movie watching 
platform, wants to improve its collection of movies and showcase 
those that are highly rated and recommend those movies to its 
customer by their movie watching footprint. For this, the company 
has collected the data and shared it with you to provide some 
analytical insights and also to come up with a recommendation 
algorithm so that it can automate its process for effective 
recommendations. The ratings are between -9 and +9.

"""
"""
Business Objectives
Minimize : Minimize less rated movies from collection of movies.
Maximize : Maximize More rated movies .
Business Constraints : Give apporiate Recommendation of similar type of movies to customer.
"""
"""
Data Dictionary

Name of feature          Type   Relevance   Description
0              Id  Quantitative  Irrelavant       User Id
1          Titles   Qualitative    Relevant  Movie Titles
2        Category   Qualitative    Relavant      Category
3         Reviews       Ordinal    Relevant       Reviews

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creating DataFrame of csv file
df=pd.read_csv("Entertainment.csv")
df

df.head
# describe - 5 number summary
df.describe()
df.shape
# 51 rows and 4 columns
df.columns
# 4 columns - 'Id', 'Titles', 'Category', 'Reviews'

# check for null values
df.isnull()
# False
df.isnull().sum()
# sum is 0 , no null values , no need to drop

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()
# most of the rating is between 2 to 4.
# id between 1000 to 5000.

# Scatter plot 

sns.set_style("whitegrid");
sns.FacetGrid(df,) \
   .map(plt.scatter, "Reviews","Category") \
   .add_legend();
plt.show();

# Displot
plt.close();
sns.set_style("whitegrid");
sns.displot(df);
plt.show()

# boxplot
# boxplot on red column
sns.boxplot(df.Reviews)
# There are 2 outliers in rating column.

# boxplot on df column
sns.boxplot(df)
# There is no outliers on columns.

# histplot
sns.histplot(df['Reviews'],kde=True)
# Reviews is not skew
# the normallly distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# mean
df.mean()
# mean of Reviews is 36.28

# median
df.median()
# median of rating is 5.92.

# standard deviation
df.std()
# std of rating is 49.03

# Data Preproccesing

df.dtypes
# Id in integer , Titles and Category in object and Reviews in float data types.

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# output is 0
# Hence there is no duplicate value in dataframe.

# Recommandation

df.Reviews
# Here we are  considering only Category column
from sklearn.feature_extraction.text import TfidfVectorizer
# This is term frequency inverse document
# Each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
# It is going to create Tfidfvectorizer to separate all stop words.

# out all words from the row
# Check is there any null value
df['Category'].isnull().sum()
#There are 0 null values

# Impute these empty spaces , game is like simple imputer
df['Category']=df['Category'].fillna('Movie')

# create tfidf_matrix Vectorizer matrix
tfidf_matrix=tfidf.fit_transform(df.Category)
tfidf_matrix.shape
# 51 rows 34 columns
# It has created sparse matrix , it means that we have 34 games 
# on this particular matrix, we want to do item based recommendation

# import linear kernel
from sklearn.metrics.pairwise import linear_kernel

# This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
# Each element of tfidf_matrix is compared
# with each element of tfidf_matrix only

# output will be similarity matrix of size 51 x 34 size
# Here in cosine_sim_matrix ,
# there are no movie names only index are provided 
# We will try to map movie name with movie index given


env_index=pd.Series(df.index,index=df['Category']).drop_duplicates()
# Here we are converting env_index into series format , we want index and corresponding
env_id=env_index['Comedy, Drama, School, Shounen, Sports']
env_id
# env id 5, 14, 43

# function
def get_recommendations(Name,topN):
    env_id=env_index[Name]
    
    # We want to capture whole row of given game
    # category , its reviews,Title and  id
    # For that purpose we are applying cosine_sin_matrix to enumerate function
    # Enumerate function create a object which we need to create in list form
    # we are using enumerate function ,
     
    cosine_scores=list(enumerate(cosine_sim_matrix[env_id]))
    # The cosine score captured , we want to arrange in descending order
    # we can recommend top 10 based on highest similarity i.e. score
    # x[0]=index and x[1] is cosine score
    # we wnat arrange tuples according to decreasing order of the score not index
    # Sorting the cosine_similarity score based on the scores i.e. x[1]
    
    cosine_scores=sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    # Get the scores of top N most similar category movies
    # To capture TopN movies ,  you need to give topN+1
    
    cosine_scores_N=cosine_scores[0: topN+1]
    # getting  the movie index
    
    env_idx=[i[0] for i in cosine_scores_N]
    # getting cosine score
    env_scores=[i[1] for i in cosine_scores_N]
    #we are going to use this information to create a daatframe
    #create a empty dataframe
    env_similar_show=pd.DataFrame(columns=['Category','score'])
    #assign anime_idx to name column 
    env_similar_show['Category']=df.loc[env_idx, 'Category']
    #assign score to score column
    env_similar_show['score']=env_scores
    #while assigning values, it is by default capturing original
    #index of the movie 
    #we want to reset the index
    env_similar_show.reset_index(inplace=True)
    print(env_similar_show)
    
#Enter your 
get_recommendations('Action, Adventure, Shounen, Super Power', 10)
# we got 10 recommendation related to this ente.
'''
   index                                           Category     score
0       6            Action, Adventure, Shounen, Super Power  1.000000
1      23  Action, Comedy, Parody, Sci-Fi, Seinen, Super ...  0.567625
2      13  Action, Drama, Mecha, Military, Sci-Fi, Super ...  0.548868
3      19  Action, Mecha, Military, School, Sci-Fi, Super...  0.534361
4      24                         Action, Adventure, Fantasy  0.421340
5       1  Action, Adventure, Drama, Fantasy, Magic, Mili...  0.320244
6      42                    Action, Comedy, School, Shounen  0.314217
7      15                     Adventure, Drama, Supernatural  0.267914
8      29           Action, Adventure, Comedy, Mecha, Sci-Fi  0.266933
9      22    Action, Adventure, Comedy, Drama, Sci-Fi, Space  0.250039
10     35                 Adventure, Drama, Fantasy, Romance  0.210673

'''
