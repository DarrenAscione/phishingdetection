from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from helper_class import KFolds
from helper_class import load_data
import time


if __name__ == '__main__':
    print "Tutorial: Training a support vector machine to detect phishing websites"
    user_mode = raw_input('Select (A) for Data Size (B) for kFolds: ')
    if user_mode == 'A':
        user_input = raw_input('Key in size of training set to train model on: ')
        print 'Testing on a single reserved set...'
        start_time = time.time()
        train_inputs, train_outputs, test_inputs, test_outputs = load_data(user_input)
        print "Training data loaded."

        svc = svm.SVC(kernel='rbf')
        print "Support Vector Machine classifier created. \n"
        # include gridSearch on defined parameters
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        classifier = GridSearchCV(svc, parameters)        
        print "Training data loaded."

        print "Beginning model training."
        # Train the support vector machine classifier
        classifier.fit(train_inputs, train_outputs)
        print "Model training completed."

        # Use the trained classifier to make predictions on the test data
        predictions = classifier.predict(test_inputs)
        print "Predictions on testing data computed."

        # Print the accuracy (percentage of phishing websites correctly predicted)
        accuracy = 100.0 * accuracy_score(test_outputs, predictions)
        print "The accuracy of your support vector machine on testing data is: " + str(accuracy) + '\n'
        print "Total time taken for algorithm to complete Hold-out run: ", time.time()-start_time
    elif user_mode == 'B':
        user_input = raw_input('key in the number of folds to run: ')
        k = int(user_input)
        total_acc = 0
        kfold_data = KFolds(k)
        start_time = time.time()

        # Create a support vector machine classifier model using scikit-learn
        svc = svm.SVC()
        print "Support Vector Machine classifier created. \n"

        # include gridSearch on defined parameters
        parameters = {'kernel':['poly'], 'C':[1, 10]}
        classifier = GridSearchCV(svc, parameters)

        for i in xrange(k):
            print "FOLD NUMBER: ", i+1, '\n'
            # Load the training data
            train_inputs = kfold_data[i]['training_inputs'] 
            train_outputs = kfold_data[i]['training_outputs'] 
            test_inputs = kfold_data[i]['testing_inputs']
            test_outputs = kfold_data[i]['testing_outputs']
            print "Training data loaded."


            print "Beginning model training."
            # Train the support vector machine classifier
            classifier.fit(train_inputs, train_outputs)
            print "Model training completed."

            # Use the trained classifier to make predictions on the test data
            predictions = classifier.predict(test_inputs)
            print "Predictions on testing data computed."

            # Print the accuracy (percentage of phishing websites correctly predicted)
            accuracy = 100.0 * accuracy_score(test_outputs, predictions)
            total_acc += accuracy
            print "The accuracy of your support vector machine on testing data is: " + str(accuracy) + '\n'
        print "Overall Accuracy after ", k, " number of runs: ", total_acc/k
        print "Total time taken for algorithm to complete KFolds run: ", time.time()-start_time

