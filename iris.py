# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:57:07 2019

@author: Bikash
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
#import mglearn
#

#
iris_dataset=load_iris()
print("Key of iris_dataset: \n{}" .format(iris_dataset.keys()))
print(iris_dataset['DESCR'])
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("shape of data: {}".format(iris_dataset['data'].shape))
#print("head of data: {}".format(iris_dataset['data'].head()))
X_train,X_test,Y_train,Y_test= train_test_split(iris_dataset['data'],
                              iris_dataset['target'],random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("Y_test shape: {}".format(Y_test.shape))
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y_train, figsize=(15, 15), 
marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8 )
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
knn.predict(X_test)
print("the accuracy of model is \n{}".format(knn.score(X_test,Y_test)))