"""
s3436993     Max Yendall
s3132392     Casey-Ann Charlesworth

Assignment 2 - Practical Data Science
Assignment due 18 May 2017
"""

from include.DocumentCollection import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def confusion_matrix(actual, predicted):
    print "Will implement later ;)"
    print confusion_matrix(actual, predicted)


def k_nearest_neighbours(collection):
    dataset = 'KatyPerry'
    count_vectorizer = CountVectorizer()
    counts = collection.extract_features(dataset, count_vectorizer)

    classifier = KNeighborsClassifier()
    targets = collection.document_map[dataset]['class'].values
    classifier.fit(counts, targets)

    examples = ["watch me dance on cam", "check out my website for cheap deals www.maxyendall.com.au"]
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print "K-Nearest-Neighbours Predictions: ", predictions


def naive_bayes(collection):
    dataset = 'KatyPerry'
    count_vectorizer = CountVectorizer()
    counts = collection.extract_features(dataset, count_vectorizer)

    classifier = MultinomialNB()
    targets = collection.document_map[dataset]['class'].values
    classifier.fit(counts, targets)

    examples = ["watch me dance on cam", "check out my website for cheap deals www.maxyendall.com.au"]
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print "Naive Bayes Predictions: ", predictions


def parse_data_collection():
    # Create new document collection
    collection = DocumentCollection()
    # Populate the document map with data frames and unique key-sets
    collection.populate_map()

    # Predict using Naive Bayes
    naive_bayes(collection)
    # Predict using K Nearest Neighbours
    k_nearest_neighbours(collection)


if __name__ == "__main__": parse_data_collection()
