from scipy import stats
from numpy import absolute
from numpy import genfromtxt
from numpy import savetxt
from sklearn import preprocessing
import numpy as np
import pandas as pd

def getAbsoluteZscore(X_train, X_test):
    scaler = preprocessing.StandardScaler().fit(X_train)

    new_X_train = scaler.transform(X_train)
    abs_X_train = absolute(new_X_train)
    print(abs_X_train.shape)
    div_X_train = np.divide(abs_X_train, 3)
    print(div_X_train.shape)


    new_X_test = scaler.transform(X_test)
    abs_X_test = absolute(new_X_test)
    div_X_test = np.divide(abs_X_test, 3)


    return div_X_train, div_X_test

def minmaxnormalization(X_train, X_test):
    mm_scaler = preprocessing.MinMaxScaler()

    new_X_train = mm_scaler.fit_transform(X_train)
    new_X_test = mm_scaler.transform(X_test)

    return new_X_train, new_X_test


train_path = "/Users/anantaa/Desktop/python/readability/feature_set/train_pp.csv"
test_path = "/Users/anantaa/Desktop/python/readability/feature_set/test_pp.csv"

def preprocess():
    train_data = genfromtxt(train_path, delimiter=',', skip_header=1)
    test_data = genfromtxt(test_path, delimiter=',', skip_header=1)

    index_train = train_data[:, [0]]
    feature_set_1_train = train_data[:, [3, 4, 7, 8, 9, 10]]
    feature_set_2_train = train_data[:, [2, 5, 6]]
    # name_train = np.reshape(name_train, newshape=[name_train.shape[0],1])

    index_test = test_data[:, [0]]
    feature_set_1_test = test_data[:, [3, 4, 7, 8, 9, 10]]
    feature_set_2_test = test_data[:, [2, 5, 6]]
    # name_train = np.reshape(name_train, newshape=[name_train.shape[0],1])

    new_feature_set_1_train,  new_feature_set_1_test = minmaxnormalization(feature_set_1_train, feature_set_1_test)
    new_feature_set_2_train, new_feature_set_2_test = getAbsoluteZscore(feature_set_2_train, feature_set_2_test)

    new_train = np.hstack((index_train, new_feature_set_1_train, new_feature_set_2_train))
    new_test = np.hstack((index_test, new_feature_set_1_test, new_feature_set_2_test))

    pd.DataFrame(new_train).to_csv("foo1.csv", header=None, index=None)
    pd.DataFrame(new_test).to_csv("foo2.csv", header=None, index=None)


preprocess()
