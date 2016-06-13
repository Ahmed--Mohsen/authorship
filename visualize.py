# -*- coding: utf-8 -*-
# encoding=utf8

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
from nltk.corpus import stopwords
import csv
import nltk
import pprint
import numpy
import string
from collections import Counter
from nltk.stem.porter import *
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from nltk import pos_tag

#"""
reader=csv.reader(open("features__l0.csv","rb"),delimiter=',')
next(reader, None) 
x=list(reader)
features = [y[0:-1] for y in x]
features = numpy.array(features).astype('float')
print len(features[0])
labels = [int(y[-1].split("_")[-1]) for y in x]
print len(labels)
#"""
#features = numpy.load("features_v.npy")
#labels = numpy.load("labels_v.npy")
#print len(set(labels))
import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA


from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
print "multidimentional scaling"

#mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
#pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
#xs, ys = pos[:, 0], pos[:, 1]

#pca = PCA(n_components=2)
#pos = pca.fit_transform(tfidf_matrix.toarray())  # shape (n_components, n_samples)
#xs, ys = pos[:, 0], pos[:, 1]

"""
svd = TruncatedSVD(n_components=2, random_state=42)
#pos = svd.fit_transform(tfidf_matrix) 
pos = svd.fit_transform(features) 
#xs, ys, zs= pos[:, 0], pos[:, 1], pos[:, 2]							 
xs, ys= pos[:, 0], pos[:, 1]
"""

xs, ys= features[:, 0], features[:, 1]
# Plot result
print "plotting"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure(figsize=(8, 3))
#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

# Black removed and is used for noise instead.
#unique_labels = set(labels)
#print labels
unique_labels = max(set(labels)) + 1
print set(labels)
print unique_labels
#print set(clusters)
colors = plt.cm.Spectral(numpy.linspace(0, 1, unique_labels))
#fig = plt.figure(1, figsize=(10, 10))
#ax = Axes3D(fig, elev=-150, azim=110)
#ax.scatter(xs,ys, zs, c=labels, cmap=plt.cm.Paired)
plt.scatter(xs,ys, c=labels, cmap=plt.cm.Paired, s=15, facecolor='0.5', lw = 0)
#plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()