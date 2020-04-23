# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:12:33 2020

@author: lucas
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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


nb = GaussianNB()

nb.fit(x,y)
print(nb.class_prior_)
labels = nb.predict(data_test)

lab_test = lab_test.as_matrix()

# Calculate the accuracy of the model
correct = 0
for i in range(len(labels)):
    if labels[i] == lab_test[i]:
        correct += 1

print("Accuracy: " + str(correct/len(labels)))