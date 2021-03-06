#!/usr/bin/env python
#!/usr/bin/env python -W ignore::DeprecationWarning

# File name: ExplorationModule.py
# Author: Max Yendall
# Course: Practical Data Science
# Date last modified: 19/05/2017
# Python Version: 2.7

from nltk.corpus import stopwords
from dateutil.parser import parse
from collections import Counter

import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


class ExplorationModule(object):

    def contains_url(self, string):
        """
        Uses regular expression matching to return all occurrences of a URL in a string
        :param string: A sentence
        :return: Count of occurrences of URLs
        """
        return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)

    def author_visualisation(self, spam_collection):
        """
        Fetch the top 5 spamming authors and visualise
        :param spam_collection: Spam YouTube Collection
        :return: PyPlot visuals
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
            group.append(spam[0])
            values.append(spam[1])
            if iterator == 0:
                break
            iterator -= 1

        y_pos = np.arange(len(group))

        plt.barh(y_pos, values, align='center', alpha=0.5)
        plt.yticks(y_pos, group)
        plt.xlabel('Number of Spam Comments')
        plt.ylabel('YouTube Author')
        plt.title('Top 5 Spamming Authors \nin YouTube Comment Corpus')

        plt.show()

    def url_visualisation(self, spam_collection):
        """
        Find URL presence within spam comments and visualise
        :param spam_collection: Spam YouTube Collection
        :return: PyPlot Visuals
        """

        spam_url_count = 0
        nonspam_url_count = 0
        for doc, content in spam_collection.iteritems():
            spam_set = content.loc[content['class'] == 'Spam']
            nonspam_set = content.loc[content['class'] == 'Not Spam']

            for index, row in spam_set.iterrows():
                if self.contains_url(row['content']):
                    spam_url_count += 1
            for index, row in nonspam_set.iterrows():
                if self.contains_url(row['content']):
                    nonspam_url_count += 1

        sizes = [spam_url_count, nonspam_url_count]
        colors = ['lightcoral', 'lightskyblue']
        explode = (0.3, 0)

        plt.title('URL Presence within Spam and Non-Spam Comments \n in YouTube Comment Corpus')
        plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.legend(['Spam', 'Not Spam'], loc='best')
        plt.axis('equal')
        plt.show()

    def check_hour_range(self, hour):
        """
        Check which stage of the day an hour falls
        :param hour: hour as a integer
        :return: String of time in day
        """
        if 0 <= hour <= 5:
            return 'Early Morning'
        if 6 <= hour <= 11:
            return 'Day Time'
        if 12 <= hour <= 17:
            return 'Afternoon'
        if 18 <= hour <= 23:
            return 'Evening'

    def time_visualisation(self, spam_collection):
        """
        Visualise the time of day when comments were placed on various YouTube videos
        :param spam_collection: Spam YouTube Collection
        :return: PyPlot visuals
        """
        # Define range hash tables
        spamtime_ranges = {'Early Morning': 0, "Day Time": 0, "Afternoon": 0, "Evening": 0}
        nonspamtime_ranges = {'Early Morning': 0, "Day Time": 0, "Afternoon": 0, "Evening": 0}

        # Iterate through each document and check when the comment was placed on the video
        for doc, content in spam_collection.iteritems():
            spam_set = content.loc[content['class'] == 'Spam']
            nonspam_set = content.loc[content['class'] == "Not Spam"]
            for index, row in spam_set.iterrows():
                date = row['date']
                if not str(date) == 'NaT':
                    hour = int(parse(str(date)).time().hour)
                    spamtime_ranges[self.check_hour_range(hour)] += 1
            for index, row in nonspam_set.iterrows():
                date = row['date']
                if not str(date) == 'NaT':
                    hour = int(parse(str(date)).time().hour)
                    nonspamtime_ranges[self.check_hour_range(hour)] += 1

        # Plot stacked bar chart showing both spam and non-spam comments categorised
        y_pos = np.arange(len(spamtime_ranges.keys()))

        plt.bar(y_pos, spamtime_ranges.values(), align='center', color='indianred', label='Spam', alpha=0.5)
        plt.bar(y_pos, nonspamtime_ranges.values(), align='center', color='darkred', label='Not Spam', alpha=0.5)
        plt.xticks(y_pos, spamtime_ranges.keys())
        plt.ylabel('Time of Day')
        plt.title('Distribution of Comments by Time of Day \n in YouTube Comment Corpus')
        plt.legend(loc='best')

        plt.show()

    def date_visualisation(self, spam_collection):
        """
        Visualise spam frequencies for each particular year captured in the corpus
        :param spam_collection: Spam YouTube Collection
        :return: PyPlot visuals
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
        plt.title('Spam Comments per Year \n in YouTube Comment Corpus')

        plt.show()

    def term_visualisation(self, spam_collection):
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
        plt.title('Most Frequent Spam Terms in YouTube Comment Corpus')
        plt.pie(values, explode=explode, labels=labels, shadow=False, startangle=140)

        plt.axis('equal')
        plt.show()

    def visualise_relationships(self, collection):
        """
        Plot various graphs to represent relationships between variables
        :param collection: YouTube Spam data-set
        :return: PyPlot visuals
        """
        author_content = {}

        for document, content in collection.document_map.iteritems():
            spam_set = content.loc[content['class'] == 'Spam']
            author_content[document] = spam_set['author']

        self.term_visualisation(collection.document_map)
        self.url_visualisation(collection.document_map)
        self.date_visualisation(collection.document_map)
        self.time_visualisation(collection.document_map)
        self.author_visualisation(author_content)

    def summarise_columns(self, collection):
        keys = collection.document_map.keys()
        frames = []
        for key in keys:
            frames.append(collection.document_map[key])
        merge = pd.concat(frames)
        print "---------------------------"
        print "Exploration Summaries"
        print "---------------------------"
        print merge['id'].describe(), "\n"
        print merge['author'].describe(), "\n"
        print merge['content'].describe(), "\n"
        print merge['date'].describe(), "\n"
        print merge['class'].describe()
        print "\n\n"
