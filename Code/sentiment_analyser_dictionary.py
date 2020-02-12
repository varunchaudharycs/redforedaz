#Data Analysis
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import nltk
from nltk.corpus import stopwords
import re
import networkx
from textblob import TextBlob

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# Reading data

test_tweets = pd.read_csv('/home/varun/Downloads/redforedaz_2018.csv', sep=';')

# Feature Engineering
def text_processing(tweet):
    # Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)

    new_tweet = form_sentence(tweet)

    # Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess

    no_punc_tweet = no_user_alpha(new_tweet)

    # Normalizing the words in tweets
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word, 'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet

    norm_tweet = ' '.join(normalization(no_punc_tweet))

    # polarity score
    def polarity(tweet):
        return TextBlob(tweet).polarity

    polarity_score = polarity(norm_tweet)

    return polarity_score

# Applying feature engineering
test_tweets['polarity'] = test_tweets['text'].apply(text_processing)

test_tweets = test_tweets[['text', 'polarity', 'date']]

test_tweets.to_csv('/home/varun/Downloads/results_2018.csv', index = False)


