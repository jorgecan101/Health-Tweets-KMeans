# -*- coding: utf-8 -*-
#author(s): Jorge Cancino

import pandas as pd
import numpy as np

#Start preprocessing steps
def preprocess(dataset):
  df = pd.read_csv(dataset, sep = "|", encoding = "UTF-8", header = None)
  #Dropping the first two columns with tweet id and timestamp
  df = df.drop(df.columns[[0,1]], axis = 1)
  #print(df)
  #Shuffling the dataset according to requirements for K-Means algorithm
  df = df.sample(frac = 1)
  #print(df)
  #Assigning tweets as the leftover column
  df.columns = ['tweets']
  #call edit tweets to clean up the tweets(remove @'s, #'s, convert to lowercase, and remove URL's)
  df = df['tweets'].apply(edit_tweets)
  #Remove missing values
  df = df.dropna(axis = 0, how = 'any')
  #print(df)
  return df

def edit_tweets(tweet):
  #Removes @ signs and URL's
  remove = filter(lambda x: x[0] != '@' and x[:7] != 'http://', tweet.split())
  #Removes # signs and converts to lowercase
  remove_rest = [w[1:].lower().strip("'") if w[0] == "#" else w.lower().strip("'") for w in remove]
  return remove_rest

#What we need to do still...

def kmeans(dataset, K, centroids):
  #Randomly selected datapoints to be used for the centroids
  dataset = dataset.sample(frac = 1).reset_index(drop=True)
  #Start by initializing centroids with random seeds (this case is only for the first iteration of assigning the data used for centroids)
  if centroids == None:
    centroids = init_kmeans(dataset, K, centroids)
  #print(centroids)
  #Where our cluseters for our tweets will be located at
  our_cluster = {i:[] for i in range(K)}
  #Clustering tweets to centroids
  for tweets in dataset:
    tweet_distance = [jaccard_distance(tweets, centroids[c]) for c in centroids]
    #Minimum distance of tweet would be it compared to others
    minimum_dist = tweet_distance.index(min(tweet_distance))
    our_cluster[minimum_dist].append(tweets)
  #Setting new centroids by calling update_centroids method
  new_centroids = update_centroids(our_cluster, K)
  #Then checks whether or not there is a convergence
  convergence = False
  #check_convergence(dataset, K, centroids, new_centroids, our_cluster)
  centroids_tweets = list(centroids.values())
  new_centroids_tweets = list(new_centroids.values())
  convergence = check_convergence(convergence, K, centroids_tweets, new_centroids_tweets)
  if convergence == False:
    #not convergered
    print("not converged...")
    centroids = new_centroids.copy()
    kmeans(dataset, K, centroids)
  else:
    #converged
    print("converged")
    sse_result = sse(our_cluster, centroids)
    #Print the Sum of Square Error Result
    print("SSE TOTAL: ", sse_result)
    #Print the cluster number and number of tweets inside each cluster
    for i in range(K):
      print("\nNumber of tweets in cluster ", str(i+1), " is " + str(len(our_cluster[i])))

def init_kmeans(dataset, K, centroids):
  #When we start off with None for centroids, it must be updated to have centroids based on the randomized dataset
  centroids = {}
  for i in range(K):
    if (dataset[i] not in list(centroids.keys())):
      centroids[i] = dataset[i]
  return centroids

def update_centroids(cluster, K): #Updates centroids
  centroids = {i:[] for i in range(K)}
  for i in cluster:
    new_cluster = cluster[i]
    distance = []
    total_dist = 0
    for j in new_cluster:
      if j != []:
        tweet_distance = [jaccard_distance(j, c) for c in new_cluster]
        total_dist = sum(tweet_distance)
        distance.append(total_dist)
    cluster_index = distance.index(min(distance))
    centroids[i] = new_cluster[cluster_index]
  return centroids

def check_convergence(convergence, K, centroids_tweets, new_centroids_tweets):
  #This will return true if here was a convergence found, which would mean that clustering is over, else return false
  for i in range(K):
    if (centroids_tweets[i] != new_centroids_tweets[i]):
      convergence = False
      break
    else:
      convergence = True
  return convergence

def jaccard_distance(a, b): 
  #Appears the idea of the jaccard distance is distance(a,b) = 1 - (intersection/union)
  a = set(a)
  b = set(b)
  #The & is used in python to symbolize intersection
  intersection = len(list((a & b)))
  #The | is used in python to symbolize union
  union = len(list((a | b)))
  #return the jaccard distance
  jdistance = 1 - (intersection/union)
  return jdistance

def sse(cluster, centroids): #Sum of Square Error 
  total = 0
  for ids in centroids.keys():
    for tweets in list(cluster[ids]):
      total += jaccard_distance(centroids[ids], tweets)**2
  return total

def main():
  #dataset = "bbchealth.txt" #this is just the first one in the zip file/can be changed if needed
  dataset = "https://raw.githubusercontent.com/jorgecan101/Health-Tweets-KMeans/main/bbchealth.txt" #same as other dataset, but with the bbchealth.txt file hosted on github
  K = 5 #Number of clusters K <----Change this depending on how many K's you want.
  centroids = None # because we're at the start? Seeds for centroids need to be random numbers???
  preprocessed_data = preprocess(dataset) #call the preprocessing
  kmeans(preprocessed_data, K, centroids) #preform the kmeans on the preprocessed data

if __name__ == "__main__":
  main()
