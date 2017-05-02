from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from settings import *
import pandas as pd
from pandas import DataFrame
import random
import numpy


class DocumentCollection:
    # Split collections by class as Hash Tables
    data_frame = pd.DataFrame
    count_vectorizer = CountVectorizer()
    # Entire set Hash Tables
    feature_set = {}
    document_map = {}
    training_set = {}
    test_set = {}

    def __init__(self):
        pass

    def extract_data(self, key, dataset):
        rows = []
        index = []
        for key, data in dataset.iteritems():
            print data
            rows.append({'content': data['content'], 'class': data['class']})
            index.append(key)

        data_frame = DataFrame(rows, index=index)
        return data_frame

    def naive_bayes(self):
        count_vectorizer = CountVectorizer()
        counts = count_vectorizer.fit_transform(self.document_map['Psy']['content'].values)

        classifier = MultinomialNB()
        targets = self.document_map['Psy']['class'].values
        classifier.fit(counts, targets)

        examples = ["Gaming channels are cool",
                    "check out my website for cheap deals www.maxyendall.com.au"]
        example_counts = count_vectorizer.transform(examples)
        predictions = classifier.predict(example_counts)
        print predictions


    def populate_map(self):
        """
        Populate a document map to store the collection of data for referencing
        :return: Populated collection hash table
        """
        # Declare constants for CSV parsing
        header = ["id", "author", "date", "content", "class"]
        # Read data into document map
        for filename in os.listdir(DATA_ROOT):
            # Ensure the file is a CSV file
            if filename.endswith(".csv"):
                # Create path and key for hash table
                file_path = DATA_ROOT + "/" + filename
                file_key = os.path.splitext(filename)[0]
                self.document_map[file_key] = pd.read_csv(file_path, sep=',', names=header, skiprows=1)

        return self.split_data()

    def print_collection(self):
        """
        Pretty print the collection of data
        :return: Pretty print to stdout
        """
        for key, value in self.document_map.iteritems():
            print "\nComment Collection for: ", key
            print value.to_string()

    def leave_one_out(self):
        """
        Select a random data frame and use that for testing
        :return:
        """
        rand_set = random.choice(self.document_map.keys())
        print self.document_map[rand_set]

    def extract_features(self):
        """
        Extract features using a count vectorizer to store word frequencies
        :return:
        """
        print self.data_frame['content'].values
        return self.count_vectorizer.fit_transform(self.data_frame['content'].values)
