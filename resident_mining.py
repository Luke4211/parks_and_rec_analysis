# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:05:04 2020

@author: lucas

Classifies whether or not residents willl
respond to the survey depending on their attributes
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Measures accuracy of models
def accuracy(predictions, target, model_name):
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == target[i]:
            correct += 1
    print("Accuracy of " + model_name + ": " + str(correct/len(labels)))
# Read residents into pandas dataframe
df = pd.read_csv("residents2.csv", names = ['neigh', 'income', 'race','hhsize','age','edu','gender','survey'])

# Split df into training data and class labels
x = pd.DataFrame(df.loc[:,'neigh':'gender'])
y = pd.DataFrame(df.loc[:,'survey'], columns = ["survey"])    
npa = df.as_matrix()

# Randomly sample from x and y, using 70% of the data to train
# Split the samples into training and test data/labels
data_train, data_test, lab_train, lab_test = train_test_split(x, y, test_size=0.3 )

# Create a new tree, and fit it with the training data
# Only allow 5 leaf nodes to prevent overfitting
# Limiting leafs also makes it easier to glean insight
tree = DecisionTreeClassifier(max_leaf_nodes = 5)
tree.fit(data_train, lab_train)

# Predict on the test data
labels = tree.predict(data_test)

lab_test = lab_test.as_matrix()


# Calculate the accuracy of the model
accuracy(labels, lab_test, "Decision Tree")

# Export tree to a .dot file, which can be later converted
# to a .png. Uncomment to export
'''
export_graphviz(tree, out_file = "tree.dot",
                feature_names = list(x.columns),
                class_names = ["no survey", "survey"],
                filled = True,
                rounded = True)
'''


'''
 Now we will see if our NaiveBayes model will outperform the decision
 tree
'''
nb = GaussianNB()

nb.fit(x,y)

labels = nb.predict(data_test)



# Calculate the accuracy of the model
accuracy(labels, lab_test, "Naive Bayes")

''' SVM
'''
sv = svm.SVC(max_iter = 5000)
sv.fit(x,y)
print("SVM accuracy: " + str(sv.score(data_test, lab_test)))

''' 
    Neural Network
'''

ann = MLPClassifier(hidden_layer_sizes=(100,))

ann.fit(data_train, lab_train)
print("ANN accuracy: " + str(ann.score(data_test, lab_test)))

''' 10 fold Cross validation '''
print("10 fold cross validation: ")
kf = KFold(n_splits=10, shuffle = True)
i = 1
tree_acc = 0
nb_acc = 0
svm_acc = 0
ann_acc = 0

for train, test in kf.split(df):
    print("Fold: " + str(i))
    i += 1
    # Grab the selected indexes from the dataset
    train = df.take(train)
    test = df.take(test)
    
    tree = DecisionTreeClassifier(max_leaf_nodes = 5)
    nb = GaussianNB()
    sv = svm.SVC(gamma = 'auto', max_iter = 5000)
    ann = MLPClassifier(hidden_layer_sizes=(100,))
    
    
    #df = pd.DataFrame(train, columns = ['neigh', 'income', 'race','hhsize','age','edu','gender','survey'])
    x_train = train.loc[:,'neigh':'gender']
    y_train = train.loc[:,'survey'] 
    
    #df = pd.DataFrame(test, columns = ['neigh', 'income', 'race','hhsize','age','edu','gender','survey'])
    x_test = test.loc[:,'neigh':'gender']
    y_test = test.loc[:,'survey']
    
    tree_fit = tree.fit(x_train,y_train)
    nb_fit = nb.fit(x_train,y_train)
    sv.fit(x_train,y_train)
    ann.fit(x_train, y_train)

    scr = tree.score(x_test, y_test)
    tree_acc += scr
    print( "Accuracy of Tree: " + str(scr))
    
    scr = nb.score(x_test, y_test)
    nb_acc += scr
    print( "Accuracy of Naive Bayes: " + str(scr))
    
    scr = sv.score(x_test, y_test)
    svm_acc += scr
    print( "Accuracy of SVM: " + str(scr))
    
    scr = ann.score(x_test, y_test)
    ann_acc += scr
    print( "Accuracy of ANN: " + str(scr))
    
    print()
print("Average Tree accuracy: " + str(tree_acc/10))
print("Average Naive Bayes accuracy: " + str(nb_acc/10))
print("Average SVM accuracy: " + str(svm_acc/10))
print("Average ANN accuracy: " + str(ann_acc/10))