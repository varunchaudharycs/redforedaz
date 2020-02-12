#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

colnames = ['label', 'id', 'date', 'query', 'user', 'tweet']
train_tweets = pd.read_csv('/home/varun/Downloads/trainingandtestdata/training_sentiment140.csv',
                           names = colnames, header = None, engine = 'python')
test_tweets = pd.read_csv('/home/varun/Downloads/redforedaz_2018.csv', sep=';')

train_tweets = train_tweets[['label', 'tweet']]
# test_tweets = test_tweets['text']

# Exploratory Data Analysis
train_tweets['length'] = train_tweets['tweet'].apply(len)
fig1 = sns.barplot('label','length',data = train_tweets,palette='PRGn')
plt.title('Average Word Length vs label')
plot = fig1.get_figure()
plot.savefig('/home/varun/Downloads/Barplot.png')

fig2 = sns.countplot(x= 'label',data = train_tweets)
plt.title('Label Counts')
plot = fig2.get_figure()
plot.savefig('/home/varun/Downloads/Countplot.png')

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

    return ' '.join(normalization(no_punc_tweet))

# Applying feature engineering
train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)
test_tweets['tweet_list'] = test_tweets['text'].apply(text_processing)

print('Negative tweets-')
print(train_tweets[train_tweets['label']==0].drop('tweet',axis=1).head())
print('Neutral tweets-')
print(train_tweets[train_tweets['label']==2].drop('tweet',axis=1).head())
print('Positive tweets-')
print(train_tweets[train_tweets['label']==4].drop('tweet',axis=1).head())


# Model Selection and Machine Learning
X = train_tweets['tweet_list']
y = train_tweets['label']
test = test_tweets['tweet_list']

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(X, y, test_size=0.2)

print("Test", label_test)
print("Train", label_train)

#Machine Learning Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

# print(classification_report(predictions,label_test))
# print ('\n')
# print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))

predictions_test = pipeline.predict(test)

print("Predictions - \n", predictions_test)
