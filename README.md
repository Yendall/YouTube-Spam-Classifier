# YouTube Spam Classification

<p align="left">
  <img src="https://github.com/Yendall/Practical-Data-Science-A2/blob/master/data/email-spam.jpg">
</p>

# Author: Max Yendall

#### Table of Contents

1. [Assumptions]- Asssumptions needing to be met to ensure code will execute
2. [Explanation of Files] - Explanation of all included files in this package
3. [Running The Scripts]
    * [Running main.py]

### Content

1. [Explanation of Files]:
      * main.py: The main module modelled off the given template. Simply run this to step through all exploration, parsing and classification
    
      * include/ClassificationModule.py: The main classification class which classifies using Naive Bayes and K-Nearest Neighbours using Pipelining (sklearn)
  
      * include/ExplorationModule.py: The main exploration class which explores statistical summaries and visualises relationships between variables
		
      * include/DocumentCollection.py: The main parsing class which filters the data and creates a collection of spam documents
	
      * include/SupportVectorMachine.py: The main SVM class for the optional extension. Includes feature extraction and classification using a Linear Kernel Support Vector Machine
		
      * include/Settings.py: A settings file to set up the environment, including NLTK redirection and root folders for data and project referencing

3. [Running the scripts]:
	In order to run all four parts of this project, simply the following command from an iPython environment:
	
	%run main.py
