"""
s3436993 - Max Yendall
Assignment 2 - Practical Data Science
"""
# !/usr/bin/env python -W ignore::DeprecationWarning

import matplotlib.pyplot as plt
from SupportVectorMachine import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class ClassificationModule(object):
    """

    """
    def plot_confusion_matrix(self, confusion_mat, title, iteration, classifier_name, directory_src):
        """

        :param confusion_mat:
        :param title:
        :param iteration:
        :param classifier_name:
        :param ngram_flag:
        :return:
        """
        image_name = "Confusion_Matrix_" + str(iteration) + ".png"
        directory = "data/Confusion_Matrices/" + classifier_name + "/" + directory_src + "/"
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

    def cross_validation(self, collection):
        """

        :param collection:
        :return:
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

        :param actual:
        :param predicted:
        :param iteration:
        :param classifier_name:
        :param ngram_flag:
        :return:
        """
        # Output Confusion Matrix
        cm = confusion_matrix(actual, predicted)
        score = f1_score(actual, predicted, pos_label="Spam")
        precision = precision_score(actual, predicted, pos_label="Spam")
        recall = recall_score(actual, predicted, pos_label="Spam")
        accuracy = accuracy_score(actual, predicted)
        # plot_confusion_matrix(cm, "Spam vs. Not Spam - Fold " + str(iteration), iteration, classifier_name, directory_src)

        return score, precision, recall, accuracy, cm

    def pipeline_predict(self, collection, pipeline, name, ngram_flag, sub_title):
        """

        :param collection:
        :param pipeline:
        :param name:
        :param ngram_flag:
        :param sub_title:
        :return:
        """
        folds = 10
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        confusion_matrices = []

        for fold in range(0, folds):
            # Split data into training and test sets
            comment_train, comment_test, class_train, class_test = self.cross_validation(collection)
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

        :param classifier:
        :param ngram_flag:
        :param tfidf_flag:
        :return:
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

        :param classifier:
        :param collection:
        :param title:
        :return:
        """
        ngram_title = "1-gram and 2-gram"
        tfidf_title = "1-gram and 2-gram & TF-IDF"
        normal_title = "Direct values"
        print "---------------------------"
        print "Classifier Results for ", title
        print "---------------------------"
        self.pipeline_predict(collection, self.build_pipeline(classifier, ngram_flag=False, tfidf_flag=False), title,
                              False,
                              normal_title)
        self.pipeline_predict(collection, self.build_pipeline(classifier, ngram_flag=True, tfidf_flag=False), title,
                              True,
                              ngram_title)
        self.pipeline_predict(collection, self.build_pipeline(classifier, ngram_flag=True, tfidf_flag=True), title,
                              True,
                              tfidf_title)
