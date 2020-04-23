# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:05:04 2020

@author: lucas

Classifies whether or not residents willl
respond to the survey depending on their attributes
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import export_graphviz

# Read residents into pandas dataframe
df = pd.read_csv("residents2.csv", names = ['neigh', 'income', 'race','hhsize','age','edu','gender','survey'])

# Split df into training data and class labels
x = pd.DataFrame(df.loc[:,'neigh':'gender'])
y = pd.DataFrame(df.loc[:,'survey'], columns = ["survey"])    
np = df.as_matrix()

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
correct = 0
for i in range(len(labels)):
    if labels[i] == lab_test[i]:
        correct += 1

print("Accuracy: " + str(correct/len(labels)))

# Export tree to a .dot file, which can be later converted
# to a .png
export_graphviz(tree, out_file = "tree.dot",
                feature_names = list(x.columns),
                class_names = ["no survey", "survey"],
                filled = True,
                rounded = True)
