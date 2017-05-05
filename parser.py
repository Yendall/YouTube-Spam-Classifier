"""
s3436993 - Max Yendall

Assignment 2 - Practical Data Science
"""

import matplotlib.pyplot as plt
from include.DocumentCollection import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score


def plot_confusion_matrix(confusion_mat, title, iteration, classifier_name):
    image_name = "Confusion_Matrix_" + str(iteration) + ".png"
    directory = "data/Confusion_Matrices/"+classifier_name+"/"
    # Set class labels
    labels = ['Spam', 'Not Spam']
    # Create a matplotlib figure
    fig = plt.figure()
    # Create subplot
    ax = fig.add_subplot(111)
    # Create greyscale confusion matrix
    cax = ax.matshow(confusion_mat, cmap=plt.cm.Greys)
    plt.title(title)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig(directory+image_name)
    #plt.show()


def cross_validation(collection):
    comment_collection = []
    class_collection = []
    for key, data in collection.document_map.iteritems():
        comment_collection.append(data['content'].values)
        class_collection.append(data['class'].values)

    x_train, x_test, y_train, y_test = train_test_split(comment_collection, class_collection, test_size=0.4)

    return x_train[1], x_test[1], y_train[1], y_test[1]


def analyse_results(actual, predicted, iteration, classifier_name):

    # Output Confusion Matrix
    cm = confusion_matrix(actual, predicted)
    score = f1_score(actual, predicted, pos_label="Spam")
    precision = precision_score(actual, predicted, pos_label="Spam")
    recall = recall_score(actual, predicted, pos_label="Spam")

    plot_confusion_matrix(cm, "Spam vs. Not Spam", iteration, classifier_name)

    return score, precision, recall


def predict(collection, classifier, name, ngrams):
    folds = 20
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for fold in range(0, folds):
        # Split data into training and test sets
        comment_train, comment_test, class_train, class_test = cross_validation(collection)

        # Vectorise the features of the training set by extract term frequencies from the comments
        if ngrams:
            count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        else:
            count_vectorizer = CountVectorizer()

        comment_train = collection.extract_features_cross(comment_train, count_vectorizer)
        classifier.fit(comment_train, class_train)

        # Vectorise the features of the test set by transforming the set
        comment_test = count_vectorizer.transform(comment_test)
        # Predict "spam" or "not spam" using the test set
        predictions = classifier.predict(comment_test)

        analysis = analyse_results(class_test, predictions, fold, name)
        f1_scores.append(analysis[0])
        precision_scores.append(analysis[1])
        recall_scores.append(analysis[2])

    f1_result = sum(f1_scores) / len(f1_scores)
    precision_result = sum(precision_scores) / len(precision_scores)
    recall_result = sum(recall_scores) / len(recall_scores)

    if ngrams:
        print "Results for --", name, "-- classifier over ", folds, " Folds (1-gram and 2-gram):"
    else:
        print "Results for --", name, "-- classifier over ", folds, " Folds:"
    print "F1 Score: ", f1_result
    print "Precision: ", precision_result
    print "Recall: ", recall_result, "\n"


def parse_data_collection():
    # Create new document collection
    collection = DocumentCollection()
    # Populate the document map with data frames and unique key-sets
    collection.populate_map()

    # Predict using MultiNomial Naive Bayes (ngram model and normal model)
    naive_bayes_classifier = MultinomialNB()
    predict(collection, naive_bayes_classifier, "MultiNomial Naive Bayes",True)
    predict(collection, naive_bayes_classifier, "MultiNomial Naive Bayes", False)

    # Predict using K Nearest Neighbours (n-gram model and normal model)
    knn_classifier = KNeighborsClassifier()
    predict(collection, knn_classifier, "K Nearest Neighbours", True)
    predict(collection, knn_classifier, "K Nearest Neighbours", False)


if __name__ == "__main__": parse_data_collection()
