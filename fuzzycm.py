from fcmeans import FCM
from numpy import genfromtxt
from numpy import savetxt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import math
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

train_path = "/Users/anantaa/Desktop/python/readability/processed_dataset_w_size/processed_pp_train_1.csv"
train_path_1 = "/Users/anantaa/Desktop/python/readability/proceesed_dataset/processed_pp_train.csv"
test_path = "/Users/anantaa/Desktop/python/readability/feature_set/test_pp.csv"



def pca(train_features):
    # train_features = StandardScaler().fit_transform(train_features)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(train_features)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['pc1', 'pc2'])

    ax = sns.scatterplot(x='pc1', y='pc2', data=principalDf)
    plt.show()

    return principalDf

def fcm(train_features):
    fcm = FCM(n_clusters=2)
    fcm.fit(train_features)

    fcm_centers = fcm.centers
    fcm_labels = fcm.u.argmax(axis=1)

    # savetxt('fcm_pp.csv', fcm_labels, fmt='%i', delimiter=',')

    return fcm_centers, fcm_labels

def feature_selection():
    train_data = genfromtxt(train_path_1, delimiter=',', skip_header=1)
    test_data = genfromtxt(test_path, delimiter=',', skip_header=1)

    train_features = train_data[:, 2:]
    test_features = train_data[:, 2:]

    """no_of_features = np.size(train_features, 1)

    set_of_features = np.empty([train_features.shape[0], 1])

    for n in range(0,8):
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
        no_of_features -= 1

    # print(set_of_features.shape)"""


    pca_features = pca(train_features)
    print(pca_features.shape)

    fcm_centers, fcm_labels = fcm(pca_features)
    score = silhouette_score(pca_features, fcm_labels)

    print(fcm_centers)
    print(fcm_labels)
    print(score)
    print(davies_bouldin_score(pca_features, fcm_labels))
    # print(completeness_score(set_of_features,fcm_labels))


def test():
    train_data = genfromtxt(train_path, delimiter=',', skip_header=1)
    test_data = genfromtxt(test_path, delimiter=',', skip_header=1)

    train_features = train_data[:, [2, 3, 4, 5, 6, 7, 9, 10]]
    pca(train_features)


feature_selection()