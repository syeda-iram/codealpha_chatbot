pip install nltk    #to install nltk library

#importing required library
import nltk

#for downloading popular resources
nltk.download("popular")
nltk.download('averaged_perceptron_tagger_eng')

import re
import os
import csv

from nltk.stem.snowball import SnowballStemmer
import random
from nltk.classify import SklearnClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd

#for displaying multiple outputs in the same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#to ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action = 'ignore' , category = DeprecationWarning)

#display all columns and rows without truncation
from IPython.display import display
pd.set_option('display.max_columns' , None)
pd.set_option('display.max_rows' , None)

def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return filtered_words

def extract_tagged(sentence):
    features = []
    for tagged_word in sentence:
        word, tag = tagged_word
        if tag == 'NN' or tag == 'VBN' or tag == 'NNS' or tag == 'VBP' or tag == 'RB'  or tag == 'VBZ' or tag == 'VBG' or tag == 'PRP' or tag == 'JJ':
            features.append(word)
    return features

lmtzr = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def extract_feature(text):
    words = preprocess(text)
    tags = nltk.pos_tag(words)
    extracted_features = extract_tagged(tags)
    stemmed_words = [stemmer.stem(x) for x in extracted_features]
    result = [lmtzr.lemmatize(x) for x in stemmed_words]
    return result

def word_feats(words):
    return dict([(word, True) for word in words])

def extract_feature_from_doc(data):
    result = []
    corpus = []
    answers = {}
    for (text, category, answer) in data:
        features = extract_feature(text)
        corpus.append(features)
        result.append((word_feats(features), category))
        answers[category] = [answer]
       
    return (result, sum(corpus, []), answers)

extract_feature_from_doc([['this is the input text from the user', 'category', 'answer to give']])

def get_content(filename):
    doc = os.path.join(filename)
    with open(doc, 'r') as content_file:
        lines = csv.reader(content_file, delimiter = '|')
        data = [x for x in lines if len(x) == 3]
        return data

filename = 'chatbot.csv'
data = get_content(filename)

features_data, corpus, answers = extract_feature_from_doc(data)

#splitting data into training and testing sets
split_ratio = 0.8

def split_dataset(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    train_split = int(data_length * split_ratio)
    return (data[:train_split]), (data[train_split:])

training_data, test_data = split_dataset(features_data, split_ratio)

#save the data
np.save('training_data', training_data)
np.save('test_data', test_data)

training_data = np.load('training_data.npy', allow_pickle = True)
test_data = np.load('test_data.npy', allow_pickle = True)

def train_using_decision_tree(training_data, test_data):
    classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff = 0.6, support_cutoff = 6)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    print('training set accuracy: ', training_set_accuracy)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    print('test set accuracy: ', test_set_accuracy)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy

dtclassifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_decision_tree(training_data, test_data)

def train_using_naive_bayes(training_data, test_data):
    classifier = nltk.NaiveBayesClassifier.train(training_data)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy

classifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_naive_bayes(training_data, test_data)
print(training_set_accuracy)
print(test_set_accuracy)
print(len(classifier.most_informative_features()))
classifier.show_most_informative_features()

classifier.classify(({'hi': True, 'option': True, 'movi': True}))

def reply(input_sentence):
    category = dtclassifier.classify(word_feats(extract_feature(input_sentence)))
    if category in answers:
        return answers[category]
    else:
        return "Sorry, I don't understand your question."

print("Hey there! Looking for a movie? Tell me what you're in the mood for, and I’ll hook you up!")
while (True):
    user_input = input()
    reply(user_input)

