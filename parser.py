"""
s3436993     Max Yendall
s3132392     Casey-Ann Charlesworth

Assignment 2 - Practical Data Science
Assignment due 18 May 2017
"""

from include.DocumentCollection import *


def parse_data_collection():

    # Create new document collection
    collection = DocumentCollection()
    # Populate the document map with data frames and unique key-sets
    data = collection.populate_map()

    print data


if __name__ == "__main__": parse_data_collection()
