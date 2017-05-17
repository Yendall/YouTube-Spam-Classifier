import operator
import matplotlib.pyplot as plt
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from include.DocumentCollection import *
from dateutil.parser import parse


def contains_url(string):
    """
    
    :param string: 
    :return: 
    """
    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)


def author_visualisation(spam_collection):
    """
    
    :param spam_collection: 
    :return: 
    """

    spam_author_collection = dict.fromkeys(spam_collection)
    for data, author_set in spam_collection.iteritems():
        for author in author_set:
            spam_author_collection[author] = 1

    for data, author_set in spam_collection.iteritems():
        for author in author_set:
            spam_author_collection[author] += 1

    spam_list = sorted(spam_author_collection.items(), key=operator.itemgetter(1))

    group = []
    values = []
    iterator = 5
    for spam in reversed(spam_list):
        print spam
        group.append(spam[0])
        values.append(spam[1])
        if iterator == 0:
            break
        iterator -= 1

    y_pos = np.arange(len(group))

    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, group)
    plt.xlabel('Number of Spam Comments')
    plt.title('Top 5 Spamming Authors')

    plt.show()


def url_visualisation(spam_collection):
    """
    
    :param spam_collection: 
    :return: 
    """

    spam_url_count = 0
    nonspam_url_count = 0
    for doc, content in spam_collection.iteritems():
        spam_set = content.loc[content['class'] == 'Spam']
        nonspam_set = content.loc[content['class'] == 'Not Spam']

        for index, row in spam_set.iterrows():
            if contains_url(row['content']):
                spam_url_count += 1
        for index, row in nonspam_set.iterrows():
            if contains_url(row['content']):
                nonspam_url_count += 1

    labels = 'Spam', 'Not Spam'
    sizes = [spam_url_count, nonspam_url_count]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0.3, 0)

    plt.title('URL Presence in Spam and Non-Spam Comments')
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)

    plt.axis('equal')
    plt.show()

def check_hour_range(hour):
    if 0 <= hour <= 5:
        return 'Early Morning'
    if 6 <= hour <= 11:
        return 'Day Time'
    if 12 <= hour <= 17:
        return 'Afternoon'
    if 18 <= hour <= 23:
        return 'Evening'

def time_visualisation(spam_collection):
    """
    
    :param spam_collection: 
    :return: 
    """
    spamtime_ranges = {'Early Morning': 0, "Day Time": 0, "Afternoon": 0, "Evening": 0}
    nonspamtime_ranges = {'Early Morning': 0, "Day Time": 0, "Afternoon": 0, "Evening": 0}

    for doc, content in spam_collection.iteritems():
        spam_set = content.loc[content['class'] == 'Spam']
        nonspam_set = content.loc[content['class'] == "Not Spam"]
        for index, row in spam_set.iterrows():
            date = row['date']
            if not str(date) == 'NaT':
                hour = int(parse(str(date)).time().hour)
                spamtime_ranges[check_hour_range(hour)] += 1
        for index, row in nonspam_set.iterrows():
            date = row['date']
            if not str(date) == 'NaT':
                hour = int(parse(str(date)).time().hour)
                nonspamtime_ranges[check_hour_range(hour)] += 1

    y_pos = np.arange(len(spamtime_ranges.keys()))

    plt.bar(y_pos, spamtime_ranges.values(), align='center', color='indianred', label='Spam', alpha=0.5)
    plt.bar(y_pos, nonspamtime_ranges.values(), align='center', color='darkred', label='Not Spam', alpha=0.5)
    plt.xticks(y_pos, spamtime_ranges.keys())
    plt.ylabel('Time of Day')
    plt.title('Distribution of Comments by Time of Day')
    plt.legend(loc='best')

    plt.show()

def date_visualisation(spam_collection):
    """
    
    :param spam_collection: 
    :return: 
    """
    yearly_spam = {'2013': 0, "2014": 0, "2015": 0}
    for doc, content in spam_collection.iteritems():
        spam_set = content.loc[content['class'] == 'Spam']
        for index, row in spam_set.iterrows():
            date = row['date']
            if not str(date) == 'NaT':
                yearly_spam[str(parse(str(date)).year)] += 1

    y_pos = np.arange(len(sorted(yearly_spam.keys())))

    plt.bar(y_pos, yearly_spam.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, sorted(yearly_spam.keys()))
    plt.ylabel('Number of Spam Comments')
    plt.title('Spam Comments per Year')

    plt.show()


def term_visualisation(spam_collection):
    """
    Create a pie chart of top 10 spam terms
    :param spam_collection: Entire collection of documents
    :return: Pie Chart visualisation
    """
    # Declare counter, storage and iteration variables
    counts = Counter()
    iterator = 10
    labels = []
    values = []
    # Grab stopwords set from NLTK
    stop = set(stopwords.words('english'))

    # Iterate through the spam collection and grab the top words from the spam content
    for doc, content in spam_collection.iteritems():
        spam_set = content.loc[content['class'] == 'Spam']
        for index, row in spam_set.iterrows():
            sentence = row['content']
            counts.update(
                word.strip('.?,!"\':<>').lower() for word in sentence.split() if word.lower().strip() not in stop)

    # Sort the dictionary of values and find the top 10 by grabbing the top 10 reversed vaues
    word_counts = sorted(counts.items(), key=operator.itemgetter(1))
    for key, val in reversed(word_counts):
        if iterator == 0:
            break
        if key != '':
            labels.append(key.decode('unicode_escape').encode('ascii', 'ignore'))
            values.append(val)
            iterator -= 1
    # Apply emphasis to the top three spam terms
    explode = (0.2, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0)
    # Plot the spam terms as a pie chart
    plt.title('Most Frequent Spam Terms')
    plt.pie(values, explode=explode, labels=labels, shadow=False, startangle=140)

    plt.axis('equal')
    plt.show()

def parse_data_collection():
    """
    
    :return: 
    """
    # Create new document collection
    collection = DocumentCollection()
    # Populate the document map with data frames and unique key-sets
    collection.populate_map()
    author_content = {}

    for document, content in collection.document_map.iteritems():
        spam_set = content.loc[content['class'] == 'Spam']
        author_content[document] = spam_set['author']

    term_visualisation(collection.document_map)
    url_visualisation(collection.document_map)
    date_visualisation(collection.document_map)
    time_visualisation(collection.document_map)
    author_visualisation(author_content)


if __name__ == "__main__": parse_data_collection()
