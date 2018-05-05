import numpy as np
import copy
import random

def load_data(value):
    """
    This helper function loads the dataset saved in the CSV file
    """


    inputs, outputs = data_gen()

    # Separate the training and testing data
    training_inputs = inputs[:int(value)]
    training_outputs = outputs[:int(value)]
    testing_inputs = inputs[8844:]
    testing_outputs = outputs[8844:]

    print '# of training input: ', len(training_inputs)
    print '# of training output: ', len(training_outputs)
    print '# of testing input: ', len(testing_inputs)
    print '# of training output: ', len(testing_outputs)
    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs

def KFolds(k=2):
    inputs, outputs = data_gen()
    print 'KFolds generating in progress...'
    #Splitting inputs && outputs into approx equal folds
    print 'Splitting input && output data into k approx equal folds...\n'
    inputs_folds = folds(inputs, k)
    outputs_folds = folds(outputs, k)
    result = []
    for i in xrange(k):
        atemp = {'testing_inputs': 0, 'training_inputs': 0}
        atemp['testing_inputs'], atemp['training_inputs'] = union(inputs_folds, i)
        atemp['testing_outputs'], atemp['training_outputs'] = union(outputs_folds, i)
        result.append(atemp)
    print '# of folds generated: ', len(result)
    print '# of training input: ', len(result[0]['training_outputs'])
    print '# of training output: ', len(result[0]['training_inputs'])
    print '# of testing input: ', len(result[0]['testing_inputs'])
    print '# of training output: ', len(result[0]['testing_outputs'])
    return result

def union(alist, i):
    train = []
    test = []
    for index in xrange(len(alist)):
        if index != i:
            train += alist[index]
        elif index == i:
            test = alist[i]
    return test, train

def data_gen():
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
    random.shuffle(training_data)
    inputs = training_data[:,:-1]
    outputs = training_data[:, -1]
    print count_labels(outputs)
    return inputs.tolist(), outputs.tolist()

def folds(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def count_labels(alist):
    count_neg = 0
    count_pos = 0
    for i in xrange(len(alist)):
        if alist[i] == 1:
            count_pos+= 1
        else:
            count_neg+= 1
    return count_neg, count_pos

