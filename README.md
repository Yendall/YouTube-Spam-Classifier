# YouTube Spam Classification
# Author: Max Yendall

#### Table of Contents

1. [Assumptions]- Asssumptions needing to be met to ensure code will execute
2. [Explanation of Files] - Explanation of all included files in this package
3. [Running The Scripts]
    * [Running main.py]

### Content

1. [Assumptions]:
    * Classes:
    This Python application uses an Object-Oriented Paradigm with the inclusion of multiple classes.
    All class files MUST BE present in order for the scripts to execute properly. All scripts will work if kept in the same directory.

    * CSV Files:
    This Python application will read and write CSV files from the data directory ONLY. There is no user input and will
    only read a files specifically named and it must be located in the data directory of this
    application. If it is not present, the Python application will fail to run.

    * iPython Execution:
    This Python application is written as standard Python scripts, which can be executed from an iPython environment
    using the %run command.
    
    * Confusion Matrix Cluster:
    This Python application outputs confusion matrices as PNG images as it is not taking a single fold of classification, 
    but rather 10 different folds. Therefore the confusion matrix is NOT outputted as part of the classification results. 
    This can be outputtd to the terminal if it is uncommented on line 88 of ClassificationModule.py

2. [Explanation of Files]:
* main.py: The main module modelled off the given template. Simply run this to step through all exploration, parsing and classification
    
* include/ClassificationModule.py: The main classification class which classifies using Naive Bayes and K-Nearest Neighbours using Pipelining (sklearn)
  
* include/ExplorationModule.py: The main exploration class which explores statistical summaries and visualises relationships between variables
		
* include/DocumentCollection.py: The main parsing class which filters the data and creates a collection of spam documents
	
* include/SupportVectorMachine.py: The main SVM class for the optional extension. Includes feature extraction and classification using a Linear Kernel Support Vector Machine
		
* include/Settings.py: A settings file to set up the environment, including NLTK redirection and root folders for data and project referencing

3. [Running the scripts]:
	In order to run all four parts of this project, simply the following command from an iPython environment:
	
	%run main.py
