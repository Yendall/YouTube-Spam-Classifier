#!/usr/bin/env python
#!/usr/bin/env python -W ignore::DeprecationWarning

# File name: ClassificationModule.py
# Author: Max Yendall
# Course: Practical Data Science
# Date last modified: 19/05/2017
# Python Version: 2.7

import os
from SupportVectorMachine import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class ClassificationModule(object):

    def plot_confusion_matrix(self, confusion_mat, title, iteration, classifier_name, directory_src):
        """
        Generates a PNG of a greyscale confusion matrix for a given generated confusion matrix of
        test data against training data accuracy
        :param confusion_mat: Generated Confusion Matrix from sklearn
        :param title: Title of the plot
        :param iteration: Fold number in K-Fold validation for title appendices
        :param classifier_name: Classifier name for directory delegation
        :return: PNG output of a confusion matrix
        """
        image_name = "Confusion_Matrix_" + str(iteration) + ".png"
        directory = "data/Confusion_Matrices/" + classifier_name + "/" + directory_src + "/"
        # Check if directory already exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

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

        plt.savefig(directory + image_name)
        plt.close('all')

    def cross_validation_split(self, collection):
        """
        Return a split training and test set for cross validation purposes
        :param collection: Spam collection (data frames)
        :return: Split data into training and test sets
        """
        comment_collection = []
        class_collection = []
        for key, data in collection.document_map.iteritems():
            comment_collection.append(data['content'].values)
            class_collection.append(data['class'].values)

        x_train, x_test, y_train, y_test = train_test_split(comment_collection, class_collection, test_size=0.4)

        return x_train[1], x_test[1], y_train[1], y_test[1]

    def analyse_results(self, actual, predicted, iteration, classifier_name, directory_src):
        """
        Calculate all statistical results for classifier predictions, including a confusion matrix
        :param actual: Actual values of the test set as a vector
        :param predicted: Predicted values of the test set as a vector
        :param iteration: Iteration of the K-Fold validation
        :param classifier_name: The classifier name for output purposes
        :return: F1 Score, Precision, Recall, Accuracy and Confusion Matrix
        """
        # Output Confusion Matrix
        cm = confusion_matrix(actual, predicted)
        score = f1_score(actual, predicted, pos_label="Spam")
        precision = precision_score(actual, predicted, pos_label="Spam")
        recall = recall_score(actual, predicted, pos_label="Spam")
        accuracy = accuracy_score(actual, predicted)
        # Commented out for efficient, uncomment if you want to write confusion matrices to file
        # plot_confusion_matrix(cm, "Spam vs. Not Spam - Fold " + str(iteration), iteration, classifier_name, directory_src)

        return score, precision, recall, accuracy, cm

    def pipeline_predict(self, collection, pipeline, name, sub_title):
        """
        Takes a generated classification pipeline and predicts the test data over 10 iterations.
        Results are returned and outputted to the user
        :param collection: Entire Spam Collection
        :param pipeline: Chained pipeline of vectorisers and classifiers
        :param name: Name of the classifier being used
        :param sub_title: Sub-title for confusion plots
        :return: Classification results for the given classifier
        """
        folds = 10
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        confusion_matrices = []

        for fold in range(0, folds):
            # Split data into training and test sets
            comment_train, comment_test, class_train, class_test = self.cross_validation_split(collection)
            pipeline.fit(comment_train, class_train)

            # Predict "spam" or "not spam" using the test set
            predictions = pipeline.predict(comment_test)
            analysis = self.analyse_results(class_test, predictions, fold, name, sub_title)
            f1_scores.append(analysis[0])
            precision_scores.append(analysis[1])
            recall_scores.append(analysis[2])
            accuracy_scores.append(analysis[3])
            confusion_matrices.append(analysis[4])

        f1_result = (sum(f1_scores) / len(f1_scores)) * 100
        precision_result = (sum(precision_scores) / len(precision_scores)) * 100
        recall_result = (sum(recall_scores) / len(recall_scores)) * 100
        classification_error = (1 - (sum(accuracy_scores) / len(accuracy_scores))) * 100

        print "Results for --", name, "-- classifier over ", folds, "Folds - ", sub_title
        print "Confusion Matrix Cluster: See Report for Cluster (too large for terminal output)"
        print "Avg K-Fold Classification Error Rate: ", classification_error, "%"
        print "Avg F1 Score: ", f1_result, "%"
        print "Avg Precision: ", precision_result, "%"
        print "Avg Recall: ", recall_result, "%", "\n"

    def build_pipeline(self, classifier, ngram_flag, tfidf_flag):
        """
        Construct a pipeline in order to use Count Vectorisation, TF-IDF Calculation and various classifiers
        :param classifier: Classifer to train and predict with
        :param ngram_flag: Boolean flag to check if n-grams are to be used in calculation
        :param tfidf_flag: Boolean flag to check if TF-IDF is to be used in calculation
        :return: Pipeline for prediction
        """
        if ngram_flag and tfidf_flag:
            return Pipeline([
                ('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                ('tfidf_transformer', TfidfTransformer()),
                ('classifier', classifier)
            ])
        elif ngram_flag and not tfidf_flag:
            return Pipeline([
                ('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                ('classifier', classifier)
            ])
        else:
            return Pipeline([
                ('vectorizer', CountVectorizer()),
                ('classifier', classifier)
            ])

    def classifier_analysis(self, classifier, collection, title):
        """
        Begin analysis of a classifier by constructing pipelines and predicting on generated test data over K-folds
        :param classifier: Classifier to be used for training and prediction
        :param collection: Entire Spam Collection
        :param title: Title of the classifier for output formatting
        :return: Classification results for the given classifier
        """
        ngram_title = "1-gram and 2-gram"
        tfidf_title = "1-gram and 2-gram & TF-IDF"
        normal_title = "Direct values"
        print "---------------------------"
        print "Classifier Results for ", title
        print "---------------------------"
        self.pipeline_predict(collection, self.build_pipeline(classifier, ngram_flag=False, tfidf_flag=False), title,
                              normal_title)
        self.pipeline_predict(collection, self.build_pipeline(classifier, ngram_flag=True, tfidf_flag=False), title,
                              ngram_title)
        self.pipeline_predict(collection, self.build_pipeline(classifier, ngram_flag=True, tfidf_flag=True), title,
                              tfidf_title)
