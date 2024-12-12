The only things that need to be changed are in the file "main.py" Here you specify which file to run through PLTR. If you uncomment lines 20-23 then the Benchmark data will be run through PLTR. If you comment out lines 25-26 then the Taiwan data will be run through PLTR. The PLTR model is on line 55 and the methods are explained above it in line 54. It is recommended to only run 'al' or 'std'. If 'al' or 'asgl' are run then remove y_prod from 55 and comment out lines 65-92. If 'comb' or 'std' are run then uncomment those lines and make sure y_prod is in line 55.

Files: main.py (The main file where new data can be read in and preprocessed, calls PLTR and the ks and pgi functions for analysis)
	analysis_functions.py (contains all the functions related to the PLTR implementation, each function is explained, only function called outside of file is PLTR which is the base function that calls all others in the file)
	ks_score.py (implantation of calculating Kolmogornov-Smirnov Statistic, called from main.py)
	PGI.py (implementation of calculating partial Gini index, called from main.py)

This was created in the PyCharm IDE, but should work in any Python compatible IDE.

GitHub repository: https://github.com/jowen005/PenalisedLogisticTreeRegression_Replication.git