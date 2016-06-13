# -*- coding: utf-8 -*-
# encoding=utf8

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
from nltk.corpus import stopwords
import csv
from nltk import * 
from nltk.corpus import *
from nltk.stem import PorterStemmer
from nltk.util import ngrams
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
from sklearn import preprocessing
from nltk import pos_tag
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn import cross_validation




def read_corpus(corpus_root="C50"):
	corpus = PlaintextCorpusReader(corpus_root, '.*txt')
	documents_names = corpus.fileids()
	data = [corpus.raw(document_name) for document_name in documents_names]
	labels = [label.split("/")[-2] for label in documents_names] #label = author id
	print "Data Size =", len(data)
	return (data, labels)

def extract_features(x, y, analyzer="char", norm=None, ngram=(1,5), lower=True, selection="chi", feature_size=5000):
	max_features = feature_size
	if selection == "chi":
		max_features = 50000 # to be filtered using chi^2
	
	# apply extraction
	tfidf = TfidfVectorizer(max_df=0.8, min_df=3, max_features=max_features, sublinear_tf=False, use_idf=False,norm=norm, analyzer=analyzer, ngram_range=ngram, lowercase=lower)	
	features = tfidf.fit_transform(x)
	
	feature_names = tfidf.get_feature_names()
	print feature_names, len(feature_names)
	
	# apply selection only if chi square as most
	# frequent is handled using the TfidfVectorizer
	if selection == "chi":
		features = select_features(features, y, feature_size)
	print "selection size =", len(features.todense()[0])
	
	return features

def select_features(x, y, k=5000):
	chi = SelectKBest(chi2, k=k)
	features = chi.fit_transform(x, y)
	return features
	
def save_features(x, y, x_file_name="features", y_file_name="labels"):
	# convert x to numpy array
	x = numpy.asarray(x.todense(), numpy.float64)
	
	# convert y to numpy array
	le = preprocessing.LabelEncoder()
	y = le.fit_transform(y)
	y = numpy.asarray(y, numpy.int32)

	#saving to features and labels
	numpy.save(x_file_name, x)
	numpy.save(y_file_name, y)

def classify(x, y):
	clf = LinearSVC()
	scores = cross_validation.cross_val_score(clf, x, y, cv=10)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return scores.mean()

def read_csv(path):
	reader=csv.reader(open(path),delimiter=',')
	next(reader, None) 
	x = list(reader)
	features = [y[0:-1] for y in x]
	features = numpy.array(features).astype('float')
	labels = [int(y[-1].split("_")[-1]) for y in x]
	return (features, labels)


print "Read Corpus"
x,y = read_corpus("C50")

print "Extract Features"
x = extract_features(x, y, ngram=(1,3), analyzer="char", selection="chi", feature_size=6000)

#print "Select Features"
#x = select_features(x, y)

print "Save Features"
save_features(x, y, "data/features_l0", "data/labels_l0")

	
#x = numpy.load("data/features_l0.npy")
#y = numpy.load("data/labels_l0.npy")
#data = read_csv("features__l0.csv")
#classify(data[0], data[1])
