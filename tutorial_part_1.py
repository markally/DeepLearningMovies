# Part 1

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data


# quoting removes double quotes, header says the first row are column labels

# movie reviews contain HTML tags

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from 

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

stopwords_set = set(stopwords.words("english"))

def strip_HTML(text):
    return BeautifulSoup(text).get_text()

def strip_punctuation(text):
    """replace characters NOT in the set of a-z A-Z with ' '."""
    return re.sub("[^a-zA-Z]", " ", text)

def remove_stop_words(word_list):
    return [w for w in word_list if w not in stopwords_set]

def clean_review(review):
    """Return a list of cleaned words without stopwords"""
    review_text = strip_HTML(review)
    letters_only = strip_punctuation(review_text)
    lowercase_word_list = letters_only.lower().split() # Tokenize words
    meaningful_words = remove_stop_words(lowercase_word_list)
    return " ".join(meaningful_words)

def main():
    print "Cleaning review...\n"
    train['clean_review'] = train['review'].map(clean_review)

    print "Creating bag of words... \n"
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer = None,
        preprocessor = None,
        stop_words = None,
        max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(train['clean_review'])
    # Convert to dense matrix
    train_data_features.toarray()

    # fit random forest
    # predict on test data
    

    # Semantria
    # Jay Filiatrault, VP business development
    # presented at the NLP conference
    # Say I heard one of his talks
    # jay@semantria.com
    # M514-771-3405
    # Lewl/ Lool from the data masters program gave me his info
