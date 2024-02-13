# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:31:00 2023

@author: Vaibhav Bhorkade

"""

import pandas as pd

anime = pd.read_csv("C:/Datasets/anime.csv", encoding='utf8')
anime.shape
#you will get 12294x7 matrix
anime.columns
anime.genre
#Here we are  considering only genre
from sklearn.feature_extraction.text import TfidfVectorizer
#This is term frequency inverse document
#Each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
#It is going to create Tfidfvectorizer to separate all stop words.
#It is going to separate
#out all words from the row
#Now let us check is there any null vaue
anime['genre'].isnull().sum()
#There are 62 null values
#Suppose one movie has got genre Drama , Romance,..
#There may be many empty spaces
#so let us impute these empty spaces , general is like simple imputer
anime['genre']=anime['genre'].fillna('general')
#Now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape
#You will get 12294,47
#It has created sparse matrix , it means 
#that we have 47 genre 
#on this particular matrix,
#we want to do item based recommendation, if a user has 
#watched Gadar , then you can recommend Shershah movie 
from sklearn.metrics.pairwise import linear_kernel
#This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#Each element of tfidf_matrix is compared
#with each element of tfidf_matrix only
#output will be similarity matrix of size 12294x12294 size
#Here in cosine_sim_matrix ,
#there are no movie names only index are provided 
#We will try to map movie name with movie index given
#for that purpose custom function is written 
anime_index=pd.Series(anime.index,index=anime['name']).drop_duplicates()
#We are converting anime_index into series format , we want index and corresponding
anime_id=anime_index['Assassins (1995)']
anime_id
def get_recommendations(Name,topN):
    #topN=10
    #Name='Assassins (1995)'
    anime_id=anime_index[Name]
    
    #We want to capture whole row of given movie
    #name , its score and column id
    #For that purpose we are applying cosine_sin_matrix to enumerate function
    #Enumerate function create a object, 
    #which we need to create in list form
    #we are using enumerate function ,
    #what enumerate does , suppose we have given
    # (2,10,15,18) , if we apply to enumerate then it will create a list
    # (0,2,   1,10,   3,15,   4,18) 
    cosine_scores=list(enumerate(cosine_sim_matrix[anime_id]))
    #The cosine score captured , we want to arrange in descending order
    #so that
    #we can recommend top 10 based on highest similarity i.e. score
    #x[0]=index and x[1] is cosine score
    #we wnat arrange tuples according to decreasing order
    #of the score not index
    #Sorting the cosine_similarity score based on the scores i.e. x[1]
    cosine_scores=sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    #Get the scores of top N most similar movies
    #To capture TopN movies ,  you need to give topN+1
    cosine_scores_N=cosine_scores[0: topN+1]
    #getting  the movie index
    anime_idx=[i[0] for i in cosine_scores_N]
    #getting cosine score
    anime_scores=[i[1] for i in cosine_scores_N]
    #we are going to use this information to create a daatframe
    #create a empty dataframe
    anime_similar_show=pd.DataFrame(columns=['name','score'])
    #assign anime_idx to name column 
    anime_similar_show['name']=anime.loc[anime_idx, 'name']
    #assign score to score column
    anime_similar_show['score']=anime_scores
    #while assigning values, it is by default capturing original
    #index of the movie 
    #we want to reset the index
    anime_similar_show.reset_index(inplace=True)
    print(anime_similar_show)
    
#Enter your anime
get_recommendations('Bad Boys (1995)', 10)
