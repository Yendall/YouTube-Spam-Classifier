from __future__ import division
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import re
import operator


class SupportVectorMachine(object):

    def plot_confusion_matrix(self, confusion_mat, title):
        """

        :param confusion_mat:
        :param title:
        :param iteration:
        :param classifier_name:
        :param ngram_flag:
        :return:
        """

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
        plt.show()
        plt.savefig("data/SVM.png")
        plt.close('all')

    def contains_url(self, string):
        return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)

    def extract_features(self, collection, spam_word_collection):
        feature_set = {}
        feature_list = []
        comment_features = []
        comment_class = []
        iterator = 0
        for document, content in collection.iteritems():

            for index, row in content.iterrows():
                comment = row['content']
                # Extract number of characters in total
                feature_set['no_chars'] = (len(comment.replace(" ", "")))
                feature_list.append(feature_set['no_chars'])

                # Extract the ratio of alphabet characters all all characters
                no_alpha = (len([c.lower() for c in comment if c.isalpha()]))
                feature_set['alpha_chars'] = no_alpha / feature_set['no_chars']
                feature_list.append(feature_set['alpha_chars'])

                # Extract the ratio of digit characters to all characters
                no_digits = (len([c.lower() for c in comment if c.isdigit()]))
                feature_set['digit_chars'] = no_digits / feature_set['no_chars']
                feature_list.append(feature_set['digit_chars'])

                # Extract the ratio of whitespace to all characters
                feature_set['whitespace_chars'] = comment.count(" ") / feature_set['no_chars']
                feature_list.append(feature_set['whitespace_chars'])

                # Extract the ratio of special characters to all characters
                no_special = (len([c.lower() for c in comment if c in set(punctuation)]))
                feature_set['special_chars'] = no_special / feature_set['no_chars']
                feature_list.append(feature_set['no_chars'])

                # Extract the number of words in total
                feature_set['no_words'] = len(comment.split())
                feature_list.append(feature_set['no_words'])

                # Extract the average word length
                feature_set['avg_word_len'] = sum(len(word) for word in comment.split()) / len(comment.split())
                feature_list.append(feature_set['avg_word_len'])

                # Extract the ratio of unique words to all words
                feature_set['unique_words'] = sum(Counter(comment.split()).values()) / feature_set['no_words']
                feature_list.append(feature_set['unique_words'])

                # Extract the number of spam words
                word_set = [word.lower() for word in comment.split()]
                feature_set['spam_words'] = len([word for word in word_set if word in spam_word_collection])
                feature_list.append(feature_set['spam_words'])

                # Extract the number of URLs
                feature_set['url_presence'] = len(self.contains_url(comment))
                feature_list.append(feature_set['url_presence'])

                # Attach class to the feature set
                if row['class'] == 'Spam':
                    class_val = 1
                else:
                    class_val = 0

                comment_features.append(feature_list)
                comment_class.append(class_val)
                iterator += 1
                feature_list = []

        return comment_features, comment_class

    def generate_top_spam_terms(self, collection):
        """
        
        :param collection: 
        :return: 
        """

        spam_counts = Counter()
        for document, content in collection.iteritems():
            spam_set = content.loc[content['class'] == 'Spam']

            for index, row in spam_set.iterrows():
                stop = set(stopwords.words('english'))
                sentence = row['content']
                spam_counts.update(
                    word.strip('.?,!"\':<>').lower() for word in sentence.split() if word.lower().strip() not in stop)

        spam_word_counts = sorted(spam_counts.items(), key=operator.itemgetter(1))

        iterator = 20
        top_spam_words = []
        for words in reversed(spam_word_counts):
            if iterator == 0:
                break
            if not words[0]:
                continue
            else:
                top_spam_words.append(words[0].strip())
                iterator -= 1

        return top_spam_words

    def train_model(self, spam_collection, title):
        """
        
        :param spam_collection: 
        :param title: 
        :return: 
        """
        spam_terms = self.generate_top_spam_terms(spam_collection.document_map)
        comment_features, class_features = self.extract_features(spam_collection.document_map, spam_terms)

        comment_features = np.array(comment_features)
        # Center to the mean and component wise scale to unit variance
        comment_features = preprocessing.scale(comment_features)
        class_features = np.array(class_features)
        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(comment_features, class_features, test_size=0.4)

        x_train = np.squeeze(np.asarray(x_train))
        y_train = np.squeeze(np.asarray(y_train))

        # Create support vector machine
        c = 1.0
        svc = svm.SVC(kernel='linear', C=c).fit(x_train, y_train)

        predicted = svc.predict(x_test)

        # score
        test_size = len(y_test)
        score = 0
        for i in range(test_size):
            if predicted[i] == y_test[i]:
                score += 1

        actual = y_test

        cm = confusion_matrix(y_test, predicted)
        score = f1_score(actual, predicted, average='weighted', pos_label=1)
        precision = precision_score(actual, predicted, pos_label=1)
        recall = recall_score(actual, predicted, pos_label=1)
        accuracy = accuracy_score(actual, predicted)

        #self.plot_confusion_matrix(cm, "Spam vs Not Spam")

        print "Results for -- Support Vector Machine --"
        print "Confusion Matrix: \n", cm
        print "Classification Error Rate: ", (1-accuracy)*100, " %"
        print "F1 Score: ", score*100, "%"
        print "Precision: ", precision*100, "%"
        print "Recall: ", recall*100, "%", "\n"

    def __init__(self):
        pass
