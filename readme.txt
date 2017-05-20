# Assignment 1: Data Cleaning and Summarising
# Max Yendall : S3436993

#### Table of Contents

1. [Assumptions]- Asssumptions needing to be met to ensure code will execute
2. [Explanation of Files] - Explanation of all included files in this package
3. [Running The Scripts]
    * [Running main.py]

### Content

1. [Assumptions]:
    Classes:
    This Python application uses an Object-Oriented Paradigm with the inclusion of multiple classes.
    All class files MUST BE present in order for the scripts to execute properly. All scripts will work if kept in the same directory.

    CSV Files:
    This Python application will read and write CSV files from the data directory ONLY. There is no user input and will
    only read a files specifically named and it must be located in the data directory of this
    application. If it is not present, the Python application will fail to run.

    iPython Execution:
    This Python application is written as standard Python scripts, which can be executed from an iPython environment
    using the %run command.
    
    Confusion Matrix Cluster:
    This Python application outputs confusion matrices as PNG images as it is not taking a single fold of classification, 
    but rather 10 different folds. Therefore the confusion matrix is NOT outputted as part of the classification results. 
    This can be outputtd to the terminal if it is uncommented on line 88 of ClassificationModule.py

2. [Explanation of Files]:
    Header.py:
        Class header for table look-ups. Essential for the functionality of task1_parser.py and task2_plotter.py
    task1_parser.py:
        Task 1 script which reads the TeachingRatings.csv file, sanitises the data and outputs to a new CSV
    task2_plotter.py:
        Task 2 script which reads the TeachingRatings_Clean.csv file and plots data as per specifications

3. [Running the scripts]:
    Both scripts are built to be run sequentially. You must run task1_parser.py BEFORE running task2_plotter.py,
    as the parser will output a new, cleaned CSV file for reference in the plotter script.

    * [Running task1_parser.py:
        This Python script will run when calling the following command from an iPython environment:
            %run task1_parser.py
    * [Running task2_plotter.py:
        This Python script will run when calling the following command from an iPython environment:
            %run task2_plotter.py

