import logging
import os
import sys
from pathlib import Path
import json
import configparser
import pandas as pd
import numpy as np
import uuid
from sqlalchemy import create_engine
import re
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from wordcloud import WordCloud, STOPWORDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve, auc
from time import time
from sklearn.model_selection import train_test_split
SEED = 2000
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier


class Generator:

    def __init__(self):
        print('Warehousing data ...')

    # Takes user input for file paths
    def user_inputs(self):
        input_path = input('Enter the input data sample file path:')
        if not Path(input_path).is_file():
            print('Invalid file path- File does not exist. Please enter again.')
            self.user_inputs()
        return input_path

    # Reads data
    def read_file(self, input_path, _delimiter, _encoding):
        print('Reading data...')
        data_frame = pd.DataFrame
        try:
            data_frame = pd.read_csv(input_path, delimiter = _delimiter, encoding = _encoding)
        except Exception as e:
            print('An exception occurred while reading file- ', input_path)
            self.safe_exit(e)
        print('Reading data... DONE.')
        return data_frame

    # Pre-processing
    @staticmethod
    def preprocess(data_frame):
        print('Pre-processing data...')
        # Converting column names to lower case
        data_frame.columns = map(str.lower, data_frame.columns)
        print('Pre-processing data... DONE.')
        return data_frame

    # Post-processing
    @staticmethod
    def postprocess(data_frame):
        print('Post-processing data...')
        # Transforming data
        pat_1 = r"(?:\@|https?\://)\S+"
        pat_2 = r'#\w+ ?'
        combined_pat = r'|'.join((pat_1, pat_2))
        www_pat = r'www.[^ ]+'
        html_tag = r'<[^>]+>'
        negations_ = {"isn't": "is not", "can't": "can not", "couldn't": "could not", "hasn't": "has not",
                      "hadn't": "had not", "won't": "will not",
                      "wouldn't": "would not", "aren't": "are not",
                      "haven't": "have not", "doesn't": "does not", "didn't": "did not",
                      "don't": "do not", "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
                      "mightn't": "might not",
                      "mustn't": "must not"}
        negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')
        tokenizer = WordPunctTokenizer()

        def data_cleaner(text):
            try:
                stripped = re.sub(combined_pat, '', text)
                stripped = re.sub(www_pat, '', stripped)
                cleantags = re.sub(html_tag, '', stripped)
                lower_case = cleantags.lower()
                neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
                letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
                tokens = tokenizer.tokenize(letters_only)
                return (" ".join(tokens)).strip()
            except:
                return 'NC'

        data_frame['text'] = data_frame['text'].apply(data_cleaner)
        # TODO: remove irrelevant tweets like 'NURSES'
        print('Post-processing data...DONE.')
        return data_frame

    # Data viz
    @staticmethod
    def data_viz(data_frame):
        print('Visualising data...')
        tweets_string = []
        for tweet in data_frame.text:
            tweets_string.append(tweet)
        tweets_string = pd.Series(tweets_string).str.cat(sep = ' ')
        wordcloud = WordCloud(width = 1600, height = 800, max_font_size = 200, background_color = 'white')\
            .generate(tweets_string)
        plt.figure(figsize = (12, 10))
        plt.imshow(wordcloud, interpolation = "bilinear")
        plt.axis("off")
        plt.show()
        print('Visualising data...DONE.')

    # Model - Ensemble classifier
    # 0 -> False, 1 -> True
    @staticmethod
    # TODO: better model?
    def build_model(train_df, test_df):
        x_train, x_validation, y_train, y_validation = train_test_split(train_df.text, train_df.sentiment,
                                                                        test_size = .2, random_state = SEED)

        def acc_summary(pipeline, x_train, y_train, x_test, y_test):
            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)

            # Compute the accuracy
            accuracy = accuracy_score(y_test, y_pred)
            # Compute the precision and recall
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            # Compute the average precision
            average_precision = average_precision_score(y_test, y_pred)

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            print('Average precision-recall score: {0:0.2f}'.format(average_precision))
            print("accuracy score: {0:.2f}%".format(accuracy * 100))
            print("-" * 80)
            return sentiment_fit, accuracy, precision, recall, average_precision, fpr, tpr

        clf1 = LogisticRegression()
        clf2 = LinearSVC()
        clf3 = MultinomialNB()
        clf4 = RidgeClassifier()
        clf5 = PassiveAggressiveClassifier()
        _fp_ = []
        _tp_ = []
        # names = ['Logistic Regression', 'Linear SVC', 'Multinomial NB', 'Ridge Classifier',
        #          'Passive Aggresive Classifier', 'Ensemble']
        # for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], names):
        #     checker_pipeline = Pipeline([
        #         ('vectorizer', TfidfVectorizer(max_features = 100000, ngram_range = (1, 3))),
        #         ('classifier', clf)
        #     ])
        #     print("Validation result for {}".format(label))
        #     print(clf)
        #     model, clf_acc, prec, rec, avg, fp, tp = acc_summary(checker_pipeline, x_train, y_train, x_validation,
        #                                                   y_validation)
        #     _fp_.append(fp)
        #     _tp_.append(tp)
        eclf = VotingClassifier(estimators = [('lr', clf1), ('svc', clf2), ('mnb', clf3), ('rcs', clf4), ('pac', clf5)],
                                voting = 'hard')
        checker_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features = 100000, ngram_range = (1, 3))),
            ('classifier', eclf)
        ])
        # print("Validation result for Ensemble Classifier")
        # print(eclf)
        model, clf_acc, prec, rec, avg, fp, tp = acc_summary(checker_pipeline, x_train, y_train,
                                                             x_validation, y_validation)
        # _fp_.append(fp)
        # _tp_.append(tp)
        test_df['sentiment'] = model.predict(test_df.text)

        return test_df

    # Saves results
    def write_to_db(self, data_frame):
        print('Writing to DB...')
        # Filtering columns according to schema
        cols_raw = ["id", "username", "text", "retweets", "favorites", "date", "geo", "permalink", "sentiment"]
        data_frame = data_frame.reindex(columns = cols_raw)
        sql_connector = create_engine('mysql+pymysql://root:0404@localhost/redforedaz')
        try:
            # Inserting data into tables
            data_frame.to_sql('tweet_info', con=sql_connector, if_exists='append', chunksize=1000, index=False)
        except Exception as e:
            print('An exception occurred while writing to db')
            self.safe_exit(e)
        print('Writing to DB... DONE.')

    @staticmethod
    def safe_exit(e):
        print('Exception-\n{}\nExiting application.'.format(e))
        sys.exit(1)


# Stores tweets data in DB
def run():
    """
The Analytics team would like to be able the query the information inside these messages with sql.
The department head has asked you to capture and warehouse this data for easy querying.
The JSON messages are in the file “data_sample.txt“.
Warehouse this data in a set of sql tables that you feel best represents the data.
Feel free to use any relevant packages for your answer.
    """
    obj = Generator()
    # TRAINING DATA
    trainset_path = '/home/varun/PycharmProjects/redForEdAZ_sentiment/code/sentiment_generator/corpus.csv'
    train_raw = obj.read_file(trainset_path, ',', 'latin-1')
    train_raw = train_raw[train_raw.Sentiment.isnull() == False]
    train_raw['Sentiment'] = train_raw['Sentiment'].map(int)
    train_raw = train_raw[train_raw['SentimentText'].isnull() == False]
    train_raw.reset_index(inplace = True)
    train_raw.rename(columns = {'SentimentText': 'text'}, inplace = True)
    train_preprocessed = obj.preprocess(train_raw)
    train_postprocessed = obj.postprocess(train_preprocessed)
    #obj.data_viz(train_postprocessed)
    # TEST DATA
    testset_path = '/home/varun/PycharmProjects/redForEdAZ_sentiment/input/redforedaz_2018.csv'
    test_raw = obj.read_file(testset_path, ';', 'utf-8')
    test_preprocessed = obj.preprocess(test_raw)
    test_postprocessed = obj.postprocess(test_preprocessed)
    #obj.data_viz(test_postprocessed)
    # MODEL
    final_df = obj.build_model(train_postprocessed, test_postprocessed)
    print(final_df.head())
    obj.write_to_db(final_df)
    print('Warehousing data ...DONE.')


if __name__ == "__main__":
    run()