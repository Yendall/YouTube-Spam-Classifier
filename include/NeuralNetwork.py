from __future__ import division
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
import re
import operator
import pprint

class NeuralNetwork:
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
                feature_set['avg_word_len'] = sum(len(word) for word in comment.split())/len(comment.split())
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

    def partial_derivative(self, value):
        derivative = value * (1.0 - value)
        return derivative

    def calculate_sigmoid(self, value):
        sigmoid_val = 1.0 / (1.0 + np.exp(-value))
        return sigmoid_val


    def init_weight_vectors(self, input_size, hidden_size):
        np.random.seed(1)
        first_weight = 2 * np.random.random((input_size, hidden_size)) - 1
        second_weight = 2 * np.random.random((hidden_size, 1)) - 1

        return first_weight, second_weight

    def train_network(self, spam_collection):
        spam_terms = self.generate_top_spam_terms(spam_collection.document_map)
        comment_features, class_features = self.extract_features(spam_collection.document_map, spam_terms)

        comment_features = np.array(comment_features)
        # Center to the mean and component wise scale to unit variance
        comment_features = preprocessing.scale(comment_features)
        class_features = np.array(class_features)
        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(comment_features, class_features, test_size=0.2)

        x_train = np.squeeze(np.asarray(x_train))
        y_train = np.squeeze(np.asarray(y_train))
        # Declare input layer size and hidden layer size. I am using four neurons in the hidden layer
        input_layer_size = len(x_train[0])
        hidden_layer_size = 4

        weight_one, weight_two = self.init_weight_vectors(input_layer_size, hidden_layer_size)

        for iteration in xrange(25000):
            # Set the input layer to the training data
            layer0 = x_train
            # Calculate sigmoid for continuous value between 0 and 1
            layer1 = self.calculate_sigmoid(np.dot(layer0, weight_one))
            layer2 = self.calculate_sigmoid(np.dot(layer1, weight_two))

            # Calculate error from prediction and begin back propagation
            layer2_error = y_train - layer2
            layer2_delta = layer2_error * self.partial_derivative(layer2)
            layer1_error = layer2_delta.dot(weight_two.T)
            layer1_delta = layer1_error * self.partial_derivative(layer1)

            # Update all weight vectors
            weight_two += layer1.T.dot(layer2_delta)
            weight_one += layer0.T.dot(layer1_delta)

        #self.back_propagate(x_train, y_train, weight_one, weight_two, 25000)

    def back_propagate(self, x_train, y_train, weight_vec_one, weight_vec_two, iterations):

        for iteration in range(0,iterations):
            # Set the input layer to the training data
            first_layer = x_train
            # Calculate sigmoid for continuous value between 0 and 1
            second_layer = 1.0 / (1.0 + np.exp(-(np.dot(first_layer, weight_vec_one))))
            third_layer = 1.0 / (1.0 + np.exp(-(np.dot(second_layer, weight_vec_two))))

            # Calculate error from prediction and begin back propagation
            third_layer_error = y_train - third_layer
            third_delta = third_layer_error * (third_layer * (1.0 - third_layer))
            second_error = third_delta.dot(weight_vec_two.T)
            second_delta = second_error * (second_layer * (1.0 - second_layer))

            # Update all weight vectors
            weight_vec_two += second_layer.T.dot(third_delta)
            weight_vec_one += first_layer.T.dot(second_delta)


    def generate_top_spam_terms(self, collection):

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


    def __init__(self):
        pass
