from sklearn_extra.cluster import KMedoids
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import genfromtxt
from numpy import savetxt
import csv
import matplotlib.pyplot as plt

train_path = "/processed_dataset_w_size/processed_pp_train_1.csv"
test_path = "/Users/anantaa/Desktop/python/readability/proceesed_dataset/processed_pp_test.csv"

def pca(train_features):
    # train_features = StandardScaler().fit_transform(train_features)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(train_features)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['pc1', 'pc2'])

    principalDf.to_csv('result.csv')
    """ax = sns.scatterplot(x='pc1', y='pc2', data=principalDf)
        plt.show()"""

    return principalDf

def fcm(train_features):
    kmedoids = KMedoids(n_clusters=3, random_state=0).fit(train_features)

    labels = kmedoids.labels_
    cluster_centre = kmedoids.cluster_centers_
    return cluster_centre, labels

def feature_selection():
    train_data = genfromtxt(train_path, delimiter=',', skip_header=1)
    test_data = genfromtxt(test_path, delimiter=',', skip_header=1)

    train_features = train_data[:, 2:]
    test_features = train_data[:, 2:]

    """no_of_features = np.size(train_features, 1)

    set_of_features = np.empty([train_features.shape[0], 1])

    for n in range(0,1):
        max_score = -math.inf
        col_num = -math.inf
        next_feature = np.empty([0, 0])

        for i in range(0, no_of_features):
            feature = train_features[:, i]
            feature = np.reshape(feature, newshape=[feature.shape[0],1])

            fcm_centers, fcm_labels = fcm(feature)
            score = silhouette_score(feature, fcm_labels)

            if max_score < score:
                next_feature = feature
                max_score = score
                col_num = i
                print(score)

        if n == 0:
            set_of_features = next_feature
        else:
            set_of_features = np.hstack((set_of_features,next_feature))

        train_features = np.delete(train_features, np.s_[col_num], axis=1)
        no_of_features -= 1"""

    # print(set_of_features.shape)

    pca_features = pca(train_features)
    print(pca_features.shape)

    fcm_centers, fcm_labels = fcm(pca_features)
    score = silhouette_score(pca_features, fcm_labels)

    print(fcm_centers)
    print(fcm_labels)
    print(score)


feature_selection()