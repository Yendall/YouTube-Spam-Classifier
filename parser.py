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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def cross_validation(collection):

    comment_collection = []
    class_collection = []
    for key, data in collection.document_map.iteritems():
        comment_collection.append(data['content'].values)
        class_collection.append(data['class'].values)

    X_train, X_test, y_train, y_test = train_test_split(
        comment_collection, class_collection, test_size=0.4, random_state=0)

    return X_train[1], X_test[1], y_train[1], y_test[1]


def build_confusion_matrix(actual, predicted):
    print confusion_matrix(actual, predicted)


def k_nearest_neighbours(collection):

    dataset = 'KatyPerry'
    count_vectorizer = CountVectorizer()
    counts = collection.extract_features(dataset, count_vectorizer)

    classifier = KNeighborsClassifier()
    targets = collection.document_map[dataset]['class'].values
    classifier.fit(counts, targets)

    examples = ["new weight loss diet", "check out my website for cheap deals www.maxyendall.com.au"]
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print "K-Nearest-Neighbours Predictions: ", predictions


def naive_bayes(collection):

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = cross_validation(collection)

    # Vectorise the features of the training set by extract term frequencies from the comments
    count_vectorizer = CountVectorizer()
    X_train = collection.extract_features_cross(X_train, count_vectorizer)

    # Use Multinomial Naive Bayes and fit the training data
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Vectorise the features of the test set by transforming the set
    X_test = count_vectorizer.transform(X_test)
    # Predict "spam" or "not spam" using the test set
    predictions = classifier.predict(X_test)

    print "Naive Bayes Predictions: ", predictions

    # Build confusion matrix to check accuracy of the classifier
    build_confusion_matrix(y_test, predictions)


def parse_data_collection():
    # Create new document collection
    collection = DocumentCollection()
    # Populate the document map with data frames and unique key-sets
    collection.populate_map()

    # Predict using Naive Bayes
    naive_bayes(collection)
    # Predict using K Nearest Neighbours
    k_nearest_neighbours(collection)

    cross_validation(collection)


if __name__ == "__main__": parse_data_collection()
