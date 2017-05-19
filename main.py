#!/usr/bin/env python
#!/usr/bin/env python -W ignore::DeprecationWarning

# File name: main.py
# Author: Max Yendall
# Course: Practical Data Science
# Date last modified: 19/05/2017
# Python Version: 2.7

from include.DocumentCollection import *
from include.ClassificationModule import *
from include.SupportVectorMachine import *
from include.ExplorationModule import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# *************************** Task 1: Data Retrieving *******************************
# Create new document collection
collection = DocumentCollection()
# Populate the document map with data frames and unique key-sets,
#  with any necessary filtering
collection.populate_map()

# *************************** Task 2: Data Exploration ******************************

# Create new Exploration Module object
exploration_module = ExplorationModule()
# Create description for each column (text data cannot be visualised well)
exploration_module.summarise_columns(collection)
# Visualise relationships between columns
exploration_module.visualise_relationships(collection)

# *************************** Task 3: Data Modelling ********************************

# Create new Classification Module object
classification_module = ClassificationModule()

# Predict using MultiNomial Naive Bayes (ngram model, normal model and TF-IDF model)
nb_classifier = MultinomialNB()

# Create new Pipelines
title = "MultiNomial Naive Bayes"
classification_module.classifier_analysis(nb_classifier, collection, title)

# Predict using K Nearest Neighbours (ngram model, normal model and TF-IDF model)
knn_classifier = KNeighborsClassifier()
# Create new Pipelines
title = "K Nearest Neighbours"
classification_module.classifier_analysis(knn_classifier, collection, title)

# *************************** Optional Extension ************************************

# Predict using a Support Vector Machine (separate feature selection)
title = "Support Vector Machine"
support_vector_machine = SupportVectorMachine()
support_vector_machine.train_model(collection, title)


# ******************************************************************************
# **************************** End of Document *********************************
# ******************************************************************************
