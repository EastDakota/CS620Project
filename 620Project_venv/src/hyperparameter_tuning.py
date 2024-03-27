import numpy as np
import math
import sys
import numpy as np
import decision_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

X_train, y_train, X_test, y_test = [],[],[],[]
def kFoldTuning():
    # This section is at least N^3,
    # Incredibly slow, 
    # but then again, it is comparing up to 100 max_depths
    # For 16 values for k_folds
    # Currently running one value for the sake of time as an example
    MIN_FOLDS = 4
    MAX_FOLDS = 15
    MAX_DEPTH = 100
    fold_results = []
    # Looping from 4 to 15 values for k-folds
    for folds in range(MIN_FOLDS, MAX_FOLDS + 1):
        kf = KFold(n_splits=folds)
        error_rates = np.zeros((folds, MAX_DEPTH - 1))
        # Loop over all of the hyperparameter options for max_depth
        for max_depth in range(1, MAX_DEPTH):
            k = 0
            # Evaluate each one K-times
            for train_index, val_index in kf.split(X_train):
                X_tr, X_val = X_train[train_index], X_train[val_index]
                y_tr, y_val = y_train[train_index], y_train[val_index]

                tree = decision_tree.DecisionTree(max_depth=max_depth)
                tree.fit(X_tr, y_tr)

                y_val_predict = tree.predict(X_val)
                error_rates[k, max_depth-1] = 1 - np.sum(y_val == y_val_predict) / y_val.size
                k += 1

    # Average across the k folds
    error_rates_avg = np.mean(error_rates, axis=0)
    error_rates_cpy = np.array(error_rates_avg)


    min_indeces = np.argsort(error_rates_cpy)[:5]
    min_vals    = [(error_rates_avg[index] * 100) for index in min_indeces]

    plt.plot(np.arange(1, MAX_DEPTH), error_rates_avg)
    plt.title("Max Depth vs error rate (K_Folds = " + str(folds) + ")")
    plt.xlabel('max depth', fontsize=16)
    plt.ylabel('Error Rate(%)',fontsize=16)
    fold_results.append([folds, min_indeces+1, min_vals])
    tmp_message = plt.show()
    plt.clf()
    return min_indeces, min_vals

def depth_parameter_tuning():
    # This code is for comparing the test performance of
    # different hyperparameter values on the test set
    
    uniq_depths = [3, 9, 8, 10, 11, 7, 4, 6, 1, 5, 90, 94, 2, 85, 86, 91, 92, 93, 95]
    opt_depth = uniq_depths[0]
    succ_rate = -1
    for depth in uniq_depths:
        test_tree = decision_tree.DecisionTree(max_depth=depth)
        test_tree.fit(X_train, y_train)
        denom = y_test.size

        predictions = test_tree.predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if predictions[i] == y_test[i]:
                correct+=1
        success_rate = (correct/denom)
        if success_rate > succ_rate:
            succ_rate = success_rate
            opt_depth = depth
        print(f"Depth = {depth} \tError rate {(1-success_rate) * 100:.5f}%")
    
    return opt_depth
'''
ALL code past here is for analysis of the tree model
'''

def print_kfold_results(fold_results):
    for fold_set in fold_results:
        print(f"Number of Folds: {fold_set[0]}")
        for i, elem in enumerate(fold_set[1]):
            print("\t" + str(i + 1) + ". Max_Depth: " + str(elem) + " Error rate: " + str("{:.2f}".format(fold_set[2][i])) + "%")
        
# This code is used to analyze the frequency of different max_depths across several
# K-fold validations
def max_depth_freq(fold_results):
    depths = []
    for fold_set in fold_results:
        for elem in fold_set[1]:
            depths.append(elem)

    uniq_depths, freq = np.unique(np.array(depths), return_counts=True)
    uniq_depths_freq = []
    for i in range(len(freq)):
        uniq_depths_freq.append((uniq_depths[i], freq[i]))

    uniq_depths_freq.sort(key=lambda x:x[1], reverse=True)
    for i in range(len(uniq_depths_freq)):
        print('{:<7s}{:>4d}{:>10s}{:<5d}'.format("Max_Depth: ", uniq_depths_freq[i][0], " Frequency: ", uniq_depths_freq[i][1]))

def get_features(node):
    if node == None:
        return []
    else:
        return get_features(node.left) + [node.split.dim] + get_features(node.right) 

def get_feature_frequency(confusion_tree):
    features_used = get_features(confusion_tree.node)
    elems, freqs = np.unique(features_used, return_counts=True)
    print("Indeces of features used: ", elems)
    plot = plt.hist(features_used, bins=range(0, 31))
    plt.axis([0, 30, 0, max(freqs)])
    useless_out_text = plt.title("Histogram of frequencies of features used")