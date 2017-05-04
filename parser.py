"""
s3436993 - Max Yendall

Assignment 2 - Practical Data Science
Assignment due 18 May 2017
"""

import matplotlib.pyplot as plt
from include.DocumentCollection import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score


def plot_confusion_matrix(confusion_mat, title):
    labels = ['Spam', 'Not Spam']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_mat, cmap=plt.cm.Greys)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def cross_validation(collection):
    comment_collection = []
    class_collection = []
    for key, data in collection.document_map.iteritems():
        comment_collection.append(data['content'].values)
        class_collection.append(data['class'].values)

    x_train, x_test, y_train, y_test = train_test_split(comment_collection, class_collection, test_size=0.4)

    return x_train[1], x_test[1], y_train[1], y_test[1]


def cross_fold(collection, classifier):
    comment_collection = []
    class_collection = []
    for key, data in collection.document_map.iteritems():
        comment_collection.append(data['content'].values)
        class_collection.append(data['class'].values)

    scores = cross_val_score(classifier, comment_collection, class_collection, cv=5)
    print scores


def analyse_results(actual, predicted):

    # Output Confusion Matrix
    cm = confusion_matrix(actual, predicted)
    score = f1_score(actual, predicted, pos_label="Spam")
    precision = precision_score(actual, predicted, pos_label="Spam")
    recall = recall_score(actual, predicted, pos_label="Spam")

    # plot_confusion_matrix(cm, "Spam vs. Not Spam")

    return score, precision, recall


def predict(collection, classifier, name):
    folds = 10
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(0, folds):
        # Split data into training and test sets
        comment_train, comment_test, class_train, class_test = cross_validation(collection)

        # Vectorise the features of the training set by extract term frequencies from the comments
        count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        comment_train = collection.extract_features_cross(comment_train, count_vectorizer)
        classifier.fit(comment_train, class_train)

        # Vectorise the features of the test set by transforming the set
        comment_test = count_vectorizer.transform(comment_test)
        # Predict "spam" or "not spam" using the test set
        predictions = classifier.predict(comment_test)

        analysis = analyse_results(class_test, predictions)
        f1_scores.append(analysis[0])
        precision_scores.append(analysis[1])
        recall_scores.append(analysis[2])

    f1_result = sum(f1_scores) / len(f1_scores)
    precision_result = sum(precision_scores) / len(precision_scores)
    recall_result = sum(recall_scores) / len(recall_scores)

    print "Results for --", name, "-- classifier over 10 Folds:"
    print "F1 Score: ", f1_result
    print "Precision: ", precision_result
    print "Recall: ", recall_result, "\n"


def parse_data_collection():
    # Create new document collection
    collection = DocumentCollection()
    # Populate the document map with data frames and unique key-sets
    collection.populate_map()

    # Predict using Naive Bayes
    naive_bayes_classifier = MultinomialNB()
    predict(collection, naive_bayes_classifier, "MultiNomial Naive Bayes")

    # Predict using K Nearest Neighbours
    knn_classifier = KNeighborsClassifier()
    predict(collection, knn_classifier, "K Nearest Neighbours")


if __name__ == "__main__": parse_data_collection()
