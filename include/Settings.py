import os
import sys
import nltk

PROJECT_ROOT = os.path.dirname(sys.modules['__main__'].__file__)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
nltk.data.path.append('data/')