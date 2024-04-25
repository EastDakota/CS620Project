import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd

def clean_data(df):
    # Dropping Lattitude, Longitude, 
    # Y_coord, X_coord, new Georeferenced, 
    # Jurisdiction code, and precinct columns
    # (all of these are descriptions of geographic location)
    
    # We would train the model to correlate arrest 
    # precincts with the boroughs over which they have jurisdiction (most of the time)
    df = df.drop(columns=['ARREST_PRECINCT', 'JURISDICTION_CODE',\
        'X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude',\
        'New Georeferenced Column', 'LAW_CODE', 'PD_CD', 'KY_CD'])
    
    ys = df['ARREST_BORO']
    xs = df.drop(columns=['ARREST_BORO'])
    
    # https://www.grepper.com/answers/289320/how+to+convert+data+type+of+a+column+in+pandas?ucard=1
    xs['ARREST_DATE'] = xs['ARREST_DATE'].apply(lambda x: x.replace('/',''))
    # Listing and shuffling police descriptions so that order is not important
    pd_descriptions = np.unique(xs['PD_DESC'])
    random.shuffle(pd_descriptions)
    
    # Listing and shuffling police descriptions so that order is not important
    OF_descriptions = np.unique(xs['OFNS_DESC'])
    random.shuffle(OF_descriptions)
    
    race_descriptions =np.unique(xs['PERP_RACE'])
    random.shuffle(race_descriptions)
    
    age_descriptions =np.unique(xs['AGE_GROUP'])
    random.shuffle(age_descriptions)
    
    xs['PD_DESC']   = xs['PD_DESC'].apply(lambda x: np.where(pd_descriptions == x)[0])
    xs['OFNS_DESC'] = xs['OFNS_DESC'].apply(lambda x: np.where(OF_descriptions == x)[0])
    xs['PERP_RACE'] = xs['PERP_RACE'].apply(lambda x: np.where(race_descriptions == x)[0])
    xs['AGE_GROUP'] = xs['AGE_GROUP'].apply(lambda x: np.where(age_descriptions == x)[0])
    # This is not me being sexist, I had to turn sexes into numbers (I'm sorry)
    xs['PERP_SEX']  = xs['PERP_SEX'].apply(lambda x: is_M(x))
    xs['LAW_CAT_CD']  = xs['LAW_CAT_CD'].apply(lambda x: is_M(x))
    return xs, ys
# M stands for both Male and Misdemeanor
def is_M(char):
    if char == "M":
        return 1
    else:
        return 0
    
def main():
    df = pd.read_csv(r"C:\Users\codyt\dev\python_venvs\Grad_School_venv\Grad-School-Work\620-Data-Science-and-Analytics\Project\Arrest_Data.csv")
    xs, ys = clean_data(df)
    # Why 42? Because it is the answer to life, the universe, and everything
    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test  = np.array(Y_test)
    # This code was used to produce an output of table 1
    # This section is at least N^3,
    # Incredibly slow, 
    # but then again, it is comparing up to 100 max_depths
    # For 16 values for k_folds
    # Currently running one value for the sake of time as an example
    MIN_FOLDS = 5
    MAX_FOLDS = 10
    MAX_DEPTH = 45
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
                y_tr, y_val = Y_train[train_index], Y_train[val_index]
                tree = DecisionTreeClassifier(max_depth=max_depth)
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
        plt.savefig('Max_depth vs error rate(K_Folds = ' + str(folds) + ')')
        plt.clf()
    
    for fold_set in fold_results:
        print(f"Number of Folds: {fold_set[0]}")
        for i, elem in enumerate(fold_set[1]):
            print("\t" + str(i + 1) + ". Max_Depth: " + str(elem) + " Error rate: " + str("{:.2f}".format(fold_set[2][i])) + "%")
    
if __name__ == "__main__":
    main()
    pass
