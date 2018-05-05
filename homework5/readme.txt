CS6923 Machine Learning
Homework 5
Shang-Hung Tsai
Chang Liu

Before you run the flight delay predictor, you will need to have Python 3 installed.
The packages we used include numpy, sklearn, and pandas.

The program takes two arguments. The first argument is the path to the training dataset,
and the second argument is the path to the testing dataset. Both files should be in csv
format.
For example:
    python3 FlightDelayPredictor.py flights_train.csv flights_test.csv

After running the program, a file named "test_outputs.csv" will be generated. This file 
contains the prediction to the testing dataset.
