import re
import random
import pandas as pd
from Settings import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


class DocumentCollection:
    # Split collections by class as Hash Tables
    data_frame = pd.DataFrame
    count_vectorizer = CountVectorizer()
    # Entire set Hash Tables
    feature_set = {}
    document_map = {}

    def __init__(self):
        pass

    def populate_map(self):
        """
        Populate a document map to store the collection of data for referencing
        :return: Populated collection hash table
        """
        # Declare constants for CSV parsing
        header = ["id", "author", "date", "content", "class"]
        dtypes = {"id": "object", "author": "object", "date": "object", "content": "object", "class": "int64"}
        # Read data into document map
        for filename in os.listdir(DATA_ROOT):
            # Ensure the file is a CSV file
            if filename.endswith(".csv"):
                # Create path and key for hash table
                file_path = DATA_ROOT + "/" + filename
                file_key = os.path.splitext(filename)[0]
                self.document_map[file_key] = pd.read_csv(file_path, sep=',', names=header, skiprows=1, dtype=dtypes,
                                                          parse_dates=[2])
                self.document_map[file_key]['class'] = \
                    self.document_map[file_key]['class'].map({1: 'Spam', 0: 'Not Spam'})

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

    def extract_features_cross(self, data, count_vectorizer):

        return count_vectorizer.fit_transform(data)

    def extract_features(self, dataset, count_vectorizer):
        """
        Extract features using a count vectorizer to store word frequencies
        :return:
        """

        return count_vectorizer.fit_transform(self.document_map[dataset]['content'].values)
