# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:13:04 2023

@author: Vaibhav Bhorkade
"""
"""
Problem Statement : 
Build a recommender system with the given data using UBCF.
This dataset is related to the video gaming industry and a survey was
conducted to build a recommendation engine so that the store can improve
the sales of its gaming DVDs. Snapshot of the dataset is given below.
Build a Recommendation Engine and suggest top selling DVDs to the 
store customers.
"""
"""
Business Objectives
Minimize : Minimize less rating video game DVDs in shop.
Maximize : Maximize More Rating of video gaming DVDs in shop.
Business Constraints : Give apporiate Recommendation of similar DVDs to customer.
"""
"""
Data Dictionary

Name of feature         Type   Relevance     Description
0          userId      Nominal  Irrelevant        user id 
1            game  Qualitative    Relevant   Name of games
2          rating      Ordinal    Relevant  rating of game

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creating DataFrame of csv file
df=pd.read_csv("game.csv")
df

df.head
# describe - 5 number summary
df.describe()
df.shape
# 5000 rows and 3 columns
df.columns
# 3 columns - 'userId', 'game', 'rating'

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
# most of the rating is between 3 to 4.
# userid between 0 to 1700.

# Displot
plt.close();
sns.set_style("whitegrid");
sns.displot(df);
plt.show()

# boxplot
# boxplot on red column
sns.boxplot(df.rating)
# There are 2 outliers in rating column.

# boxplot on df column
sns.boxplot(df)
# There is outliers on rating columns only.

# histplot
sns.histplot(df['rating'],kde=True)
# rating is not skew
# the normallly distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# mean
df.mean()
# mean of rating is 3.59

# median
df.median()
# median of rating is 4.0.

# standard deviation
df.std()
# std of rating is 0.9949

# Data Preproccesing

df.dtypes
# userId in integer game in object and ratings in float data types.

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

# We found many outliers in rating columns 
# Outliers treatment
# Let apply winsorizer technique for outliers tratment on rating columns
# import Winsorizer 
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['rating']
                  )
# creating new dataframe without outliers
df_t=winsor.fit_transform(df[['rating']])

# Check privious with outliers
sns.boxplot(df[['rating']])

# Now , Check the data columns without outliers
sns.boxplot(df_t['rating'])
# Outliers are removed

# Now treatement for skiwed data to make normalize
# There is scale diffrence between among the columns hence 
# normalize it
df.columns
# whenever there is mixed data apply normalization
# userId is unwanted data column hence drop it 
df=df.drop(['userId'],axis=1)
# column deleted
df.head()

# before normalize do one hot encoding
# for converting game into a dammies

df1=pd.get_dummies(df)
df1.head
# Now game data column is converted into the 0 and 1.
# and separed by columnwize.

# Normalize the data using norm function
# function 
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# Apply the norm_fun to data 
df2=norm_fun(df1.iloc[:,:])

info=df2.describe()
info
# Now data is normalize.

df.rating
# Here we are  considering only game and rating
from sklearn.feature_extraction.text import TfidfVectorizer
# This is term frequency inverse document
# Each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
# It is going to create Tfidfvectorizer to separate all stop words.

# out all words from the row
# Check is there any null value
df['game'].isnull().sum()
#There are 0 null values
#Suppose one game has gome ..

# Impute these empty spaces , game is like simple imputer
df['game']=df['game'].fillna('game')

# create tfidf_matrix Vectorizer matrix
tfidf_matrix=tfidf.fit_transform(df.game)
tfidf_matrix.shape
# 5000 rows 3068 columns
# It has created sparse matrix , it means that we have 3066 games 
# on this particular matrix, we want to do item based recommendation

# import linear kernel
from sklearn.metrics.pairwise import linear_kernel

# This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
# Each element of tfidf_matrix is compared
# with each element of tfidf_matrix only
# output will be similarity matrix of size 5000 x 3068 size
# Here in cosine_sim_matrix ,
# there are no movie names only index are provided 
# We will try to map movie name with movie index given


game_index=pd.Series(df.index,index=df['game']).drop_duplicates()
# Here we are converting game_index into series format , we want index and corresponding
game_id=game_index['Super Mario Odyssey']
game_id
# game_id is 17

# function
def get_recommendations(game,topN):
    game_id=game_index[game]
    
    # We want to capture whole row of given game
    # game , its rating and column id
    # For that purpose we are applying cosine_sin_matrix to enumerate function
    # Enumerate function create a object which we need to create in list form
    # we are using enumerate function ,
     
    cosine_scores=list(enumerate(cosine_sim_matrix[game_id]))
    # The cosine score captured , we want to arrange in descending order
    # we can recommend top 10 based on highest similarity i.e. score
    # x[0]=index and x[1] is cosine score
    # we wnat arrange tuples according to decreasing order of the score not index
    # Sorting the cosine_similarity score based on the scores i.e. x[1]
    
    cosine_scores=sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    # Get the scores of top N most similar games
    # To capture TopN game ,  you need to give topN+1
    
    cosine_scores_N=cosine_scores[0: topN+1]
    # getting  the game index
    
    game_idx=[i[0] for i in cosine_scores_N]
    # getting cosine score
    game_scores=[i[1] for i in cosine_scores_N]
    #we are going to use this information to create a daatframe
    #create a empty dataframe
    game_similar_show=pd.DataFrame(columns=['game','score'])
    #assign anime_idx to name column 
    game_similar_show['game']=df.loc[game_idx, 'game']
    #assign score to score column
    game_similar_show['score']=game_scores
    #while assigning values, it is by default capturing original
    #index of the game
    #we want to reset the index
    game_similar_show.reset_index(inplace=True)
    print(game_similar_show)
    
#Enter your 
get_recommendations('Super Mario Odyssey', 10)
# we got 10 recommendation related to this game
'''
 index                                        game     score
0      17                         Super Mario Odyssey  1.000000
1     202    Super Mario World: Super Mario Advance 2  0.645617
2      90  Super Mario Advance 4: Super Mario Bros. 3  0.621844
3    1856                         Super Mario Advance  0.564288
4     106                              Super Mario 64  0.544358
5       5                          Super Mario Galaxy  0.517662
6       6                        Super Mario Galaxy 2  0.517662
7    1521                           Super Paper Mario  0.498881
8     122                        Super Mario 3D World  0.495509
9    4377                                Lost Odyssey  0.489144
10    211                        Super Mario Sunshine  0.476687
'''
