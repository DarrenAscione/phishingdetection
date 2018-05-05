from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score

import numpy as np
import time
from helper_class import KFolds
from helper_class import load_data

if __name__ == '__main__':
    print "50.570 Project: Training a logistic regression to detect phishing websites"
    user_mode = raw_input('Select (A) for Data Size (B) for kFolds: ')
    if user_mode == 'A' or user_mode == 'a':
        user_input = raw_input('Key in size of training set to train model on: ')
        print 'Testing on a single reserved set...'
        start_time = time.time()
        train_inputs, train_outputs, test_inputs, test_outputs = load_data(user_input)
        print "Training data loaded."
         # Create a logistic regression classifier model using scikit-learn
        classifier = lr()
        print "Logistic regression classifier created."

        print "Beginning model training."
        # Train the logistic regression classifier
        classifier.fit(train_inputs, train_outputs)
        print "Model training completed."

        # Use the trained classifier to make predictions on the test data
        predictions = classifier.predict(test_inputs)
        print "Predictions on testing data computed."

        # Print the accuracy (percentage of phishing websites correctly predicted)
        accuracy = 100.0 * accuracy_score(test_outputs, predictions)
        print "The accuracy of your logistic regression on testing data is: " + str(accuracy) + '\n'
        print "Total time taken for algorithm to complete Hold Out run: ", time.time()-start_time

    elif user_mode == 'B':
        user_input = raw_input('key in the number of folds to run: ')
        k = int(user_input)
        total_acc = 0
        kfold_data = KFolds(k)
        start_time = time.time()

        for i in xrange(k):
            print "FOLD NUMBER: ", i, '\n'
            # Load the training data
            train_inputs = kfold_data[i]['training_inputs'] 
            train_outputs = kfold_data[i]['training_outputs'] 
            test_inputs = kfold_data[i]['testing_inputs']
            test_outputs = kfold_data[i]['testing_outputs']
            print "Training data loaded."

            # Create a logistic regression classifier model using scikit-learn
            classifier = lr()
            print "Logistic regression classifier created."

            print "Beginning model training."
            # Train the logistic regression classifier
            classifier.fit(train_inputs, train_outputs)
            print "Model training completed."

            # Use the trained classifier to make predictions on the test data
            predictions = classifier.predict(test_inputs)
            print "Predictions on testing data computed."

            # Print the accuracy (percentage of phishing websites correctly predicted)
            accuracy = 100.0 * accuracy_score(test_outputs, predictions)
            total_acc += accuracy
            print "The accuracy of your logistic regression on testing data is: " + str(accuracy) + '\n'
        print "Overall Accuracy after ", k, " number of runs: ", total_acc/k
        print "Total time taken for algorithm to complete KFolds run: ", time.time()-start_time


