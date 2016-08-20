# -*- coding: utf-8 -*-
# encoding=utf8

import sys  
import os
import os.path
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
from sklearn import preprocessing
import scipy.sparse as sps
from sklearn import metrics
from splitter import Splitter

# set log output
#f = open("output.log", 'w')
#sys.stdout = f


def read_corpus(corpus_root="C50"):
	corpus = PlaintextCorpusReader(corpus_root, '.*txt')
	documents_names = corpus.fileids()
	#data = [corpus.raw(document_name).decode('utf-8', 'ignore') for document_name in documents_names]
	data = [open(corpus_root+"/"+document_name).read().decode('utf-8', 'ignore') for document_name in documents_names]

	labels = ["/".join(label.split("/")[0:2]) for label in documents_names] #label = domain/author-id
	print "Data Size =", len(data)
	return (data, labels)

def extract_features(x, y, analyzer="char", norm=None, ngram=(1,5), lower=True, selection="chi", feature_size=5000):
	max_features = feature_size
	if selection == "chi":
		max_features = 20000 # to be filtered using chi^2
		
	# check min max norm
	scaling = norm
	if norm == "minmax":
		norm = None
		
	# apply extraction
	tfidf = TfidfVectorizer(max_df=0.8, min_df=3, max_features=max_features, sublinear_tf=False, use_idf=False,norm=norm, analyzer=analyzer, ngram_range=ngram, lowercase=lower)	
	features = tfidf.fit_transform(x)
	
	# apply min max scaling
	if scaling == "minmax":
		min_max_scaler = preprocessing.MinMaxScaler()
		features = min_max_scaler.fit_transform(features.todense())
	
	# apply selection only if chi square as most
	# frequent is handled using the TfidfVectorizer
	if selection == "chi":
		features = select_features(features, y, feature_size)
	
	return features

def select_features(x, y, k=5000):
	chi = SelectKBest(chi2, k=k)
	features = chi.fit_transform(x, y)
	return features
	
def save_features(x, y, x_file_name="features", y_file_name="labels"):
	# convert x to numpy array
	if sps.isspmatrix_csr(x):
		x = numpy.asarray(x.todense(), numpy.float64)
	
	# parse domain/author pairs
	#domains = [label.split("/")[0] for label in y]
	authors = [label.split("/")[-1] for label in y]
	
	# save domain for each author
	file = open(y_file_name+"-domain.csv",'w')
	file.writelines(["%s\n" % label for label in y])
	file.close()
	
	# convert y to numpy array
	le = preprocessing.LabelEncoder()
	y = le.fit_transform(authors)
	y = numpy.asarray(y, numpy.int32)
	
	#saving to features and labels
	numpy.save(x_file_name, x)
	numpy.save(y_file_name, y)
	

def classify(x, y):
	clf = LinearSVC()
	scores = cross_validation.cross_val_score(clf, x, y, cv=10)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return scores.mean()
	

def save_results(src_topic, dest_topic, accuracy, feature_size, experiment):
	file = open('shallow-cross.csv','a')
	file.write("%s,%s,%s,%d,%f\n" %(src_topic, dest_topic, experiment, feature_size, accuracy))
	file.close()

def read_csv(path):
	reader=csv.reader(open(path),delimiter=',')
	next(reader, None) 
	x = list(reader)
	features = [y[0:-1] for y in x]
	features = numpy.array(features).astype('float')
	labels = [int(y[-1].split("_")[-1]) for y in x]
	return (features, labels)


def create_dataset(corpus_root, folder_name ,analyzer, norm, ngram, lower, selection):

	# check if folder name exists or not
	directory = "data-cross/%s" %(folder_name)
	if not os.path.exists(directory):
		os.makedirs(directory)
	
	# parse the corpus
	x,y = read_corpus(corpus_root)

	# vary the features size from 1000 till 10,000
	for feature_size in range(1000, 10001, 1000):
		print "creating features of size", feature_size, "for", folder_name
		
		# check if current features have been created before
		if os.path.isfile("%s/features_%d.npy" % (directory, feature_size)):
			print " Features exist skipping..."
			continue
		features = extract_features(x, y, analyzer=analyzer, norm=norm, ngram=ngram, lower=lower, selection=selection, feature_size=feature_size)
		
		s = Splitter(features, y)
		splits = s.split()
		for domain_a, domain_b, accuracy in splits:
			save_results(domain_a, domain_b, accuracy, feature_size, folder_name)
			
		# save the features to be fed to SDA
		save_features(features, y, "%s/features_%d" % (directory, feature_size), "%s/labels_%d" % (directory, feature_size))
	

def create_char_dataset():
	# seperate n-gram most frequent
	for n in range(3, 6):
		create_dataset("C50", "%d-gram-most" %(n), "char", None, (n,n), False, "most")
		
	# seperate n-gram most chi
	for n in range(3, 6):
		create_dataset("C50", "%d-gram-chi" %(n), "char", None, (n,n), False, "chi")
	
	for lowercase in [False, True]:
		# variable n-gram
		create_dataset("C50", "1_5-gram-most-%s"%(lowercase), "char", None, (1,5), lowercase, "most")
		create_dataset("C50", "1_5-gram-chi-%s"%(lowercase), "char", None, (1,5), lowercase, "chi")
	
		# variable n-gram with normalization
		create_dataset("C50", "1_5-gram-l2-most-%s"%(lowercase), "char", "l2", (1,5), lowercase, "most")
		create_dataset("C50", "1_5-gram-l2-chi-%s"%(lowercase), "char", "l2", (1,5), lowercase, "chi")
		
		create_dataset("C50", "1_5-gram-l1-most-%s"%(lowercase), "char", "l1", (1,5), lowercase, "most")
		create_dataset("C50", "1_5-gram-l1-chi-%s"%(lowercase), "char", "l1", (1,5), lowercase, "chi")
			
		create_dataset("C50", "1_5-gram-minmax-most-%s"%(lowercase), "char", "minmax", (1,5), lowercase, "most")
		create_dataset("C50", "1_5-gram-minmax-chi-%s"%(lowercase), "char", "minmax", (1,5), lowercase, "chi")

def create_word_dataset():
	
	# bag of words
	create_dataset("C50", "words-chi", "word", None, (1,1), True, "chi")
	create_dataset("C50", "words-most", "word", None, (1,1), True, "most")
	
	# normalizations
	create_dataset("C50", "words-chi-l2", "word", "l2", (1,1), True, "chi")
	create_dataset("C50", "words-most-l2", "word", "l2", (1,1), True, "most")
	
	create_dataset("C50", "words-chi-minmax", "word", "minmax", (1,1), True, "chi")
	create_dataset("C50", "words-most-minmax", "word", "minmax", (1,1), True, "most")
	
	
def create_datasets():
	create_char_dataset()
	create_word_dataset()
	
def create_cross_dataset():
	# 3-gram
	create_dataset("guard", "%d-gram-most" %(3), "char", None, (3,3), False, "most")
	
	# frequent words
	create_dataset("guard", "words-most", "word", None, (1,1), True, "most")
	
	# [1-5] gram
	create_dataset("guard", "1_5-gram-most", "char", None, (1,5), False, "most")
	create_dataset("guard", "1_5-gram-chi", "char", None, (1,5), False, "chi")
	
	# [1-5] gram normalized
	create_dataset("guard", "1_5-gram-minmax-most", "char", "minmax", (1,5), False, "most")
	create_dataset("guard", "1_5-gram-minmax-chi", "char", "minmax", (1,5), False, "chi")

create_cross_dataset()
	
#create_datasets()
#create_dataset("guard", "3-gram-most-%s"%(False), "char", None, (1,5), False, "most")
#n = 3
#create_dataset("guard", "%d-gram-most" %(n), "char", None, (n,n), False, "most")
#x, y = read_corpus("guard")
#print y

# closing log file
#f.close()

#print "Read Corpus"
#x,y = read_corpus("C50")

#print "Extract Features"
#x = extract_features(x, y, ngram=(1,3), analyzer="char", selection="chi", feature_size=6000)

#print "Select Features"
#x = select_features(x, y)

#print "Save Features"
#save_features(x, y, "data/features_l0", "data/labels_l0")

	
#x = numpy.load("data/features_l0.npy")
#y = numpy.load("data/labels_l0.npy")
#data = read_csv("features__l0.csv")
#classify(data[0], data[1])

