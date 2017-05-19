#!/usr/bin/env python
#!/usr/bin/env python -W ignore::DeprecationWarning

# File name: Settings.py
# Author: Max Yendall
# Course: Practical Data Science
# Date last modified: 19/05/2017
# Python Version: 2.7

import os
import sys
import nltk

# Set project root
PROJECT_ROOT = os.path.dirname(sys.modules['__main__'].__file__)
# Set data folder root
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
# Set NLTK redirection for use of stopword corpora
nltk.data.path.append('data/')