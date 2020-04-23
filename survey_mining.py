# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:36:42 2020

@author: lucas
"""
#TODO: Use survey.csv to relate the meanings of the clusters
# Sum the "important" attributes in each cluster so we can
# figure out what the plot means

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as num
from numpy import savetxt
# Used to analyze which clusters prefer which 
# park attributes.
# Returns a list containing n lists, where n
# is the number of clusters.
# Each sublist contains 11 integers, which count
# how many members of the group selected each attribute
# , and another list containing
# the size of each cluster (for relative comparison)
def analyze_clusters(predictions, original, num_cluster=5):

    rtn = [[]] * num_cluster
    size_map = [0] * num_cluster
    for i in range(len(predictions)):
        if len(rtn[predictions[i]]) == 0:
            rtn[predictions[i]] = original[i]
        else:
            for j in range(len(rtn[predictions[i]])):
                rtn[predictions[i]][j] += original[i][j]
        size_map[predictions[i]] += 1
        
    return [rtn, size_map]


# Read data to dataframe
df = pd.read_csv("survey.csv", names = ["playground","courts","dogs","track","gardens","paths","pool","woods","field","picnicArea","natureArea"])

np = df.as_matrix()

km = KMeans(n_clusters = 5)

# Reduce the data into a lower dimensionality, so we can
# visualize it in 2D
reduced_data = PCA(n_components=2).fit_transform(np)


pred = km.fit_predict(np)


'''
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=test)
plt.show()
'''

# Plot the data, coloring each cluster seperately
fig, ax = plt.subplots()
scatter_x = reduced_data[:, 0]
scatter_y = reduced_data[:,1]
group = pred

arr, uni = num.unique(group, return_index=True)
print(arr)
print(uni)

for g in num.unique(group):
    i = num.where(group == g)
    ax.scatter(scatter_x[i], scatter_y[i], label=g)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * .9, box.height * .9])
ax.legend(loc='center_left', bbox_to_anchor=(1,0.5))
plt.figure(figsize=(20,20))
plt.show()

# This allows us to look deeper into what the clusters mean,
# by providing the summed important attributes for each cluster
att_freq, sizes = analyze_clusters(pred, np)

for i in range(len(att_freq)):
    savetxt("clus_" + str(i) + "_values.csv", att_freq[i], delimiter=',', fmt='%d')

savetxt("sizes.csv", sizes, delimiter=',')
    

# cls 0: 8, 2, 3, 4
# cls 1: 10
# cls 2: 1, 7
# cls 3: 9, 130
# cls 4: 6, 11

# cls 0: woods, courts, dogs, track
# cls 1: picnicArea
# cls 2: playground, pool
# cls 3: field
# cls 4: paths, natureArea