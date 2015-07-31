from nltk import * 
from nltk.corpus import *
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from string import ascii_lowercase
import os
from itertools import groupby
import math
import string
import csv
import scipy.io
import numpy as np
import random
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class AuthorIdentification:
	NGRAM_MIN = 1
	NGRAM_MAX = 5
	
	def __init__(self, corpus_root="corpus"):
		self.corpus_root = corpus_root
		self.corpus = PlaintextCorpusReader(self.corpus_root, '.*txt')
		self.documents_names = self.corpus.fileids()
		self.labels = [label.split("/")[-2] for label in self.documents_names] #label = author id
		self.special_chars = ['~', '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '|']
		self.punctuations = [punc for punc in list(string.punctuation) if not punc in self.special_chars ]
		self.function_words = open("function_words.txt", 'r').read().split()
		self.pos_tags = ["ADJ", "ADV", "CNJ", "DET", "EX", "FW", "MOD", "N", "NP", "NUM", "PR", "P", "TO", "UH", "V", "VB", "VD", "VG", "VN", "WH", "JJ", "CC"]
		self.stemmer = PorterStemmer()
		self.stopwords = stopwords.words('english')
		self.document_features = []
		
		#finding ngrams to be used
		most_frequent = 20000
		raw = self.corpus.raw()
		
		
		#self.all_chars = [char for char in self.corpus.raw()]
		self.all_chars = ['*' if char.isdigit() else char.lower() for char in self.corpus.raw()]
		self.ngrams = {}
		for n in range(self.NGRAM_MIN, self.NGRAM_MAX + 1):
			print "calc %d-gram..."%(n),
			#self.ngrams.extend(ngrams(self.all_chars, n))
			ngrams_list = ngrams(self.all_chars, n)
			ngrams_dist = FreqDist(ngrams_list)
			#self.ngrams[n] = set([ngram for ngram in ngrams_list if ngrams_dist[ngram] > 1])
			self.ngrams[n] = ngrams_dist.keys()
			
			#print self.ngrams[n]
			#self.ngrams[n] = ngrams_dist.keys()[:most_frequent]
			#print len(self.ngrams)
			#self.ngrams[n] = ngrams_dist.keys()[:most_frequent]
			print len(self.ngrams[n])
		print "selecting top features..."
		#ngrams_dist = FreqDist(self.ngrams)
		#self.ngrams = ngrams_dist.keys()[:most_frequent]
		#self.ngrams = set([ngram for ngram in self.ngrams if ngrams_dist[ngram] > 2])
		#print len(self.ngrams)
		#print self.ngrams
		
		#finding the words to be used for bag of words representation of the document		
		self.all_words = ([ self.stemmer.stem(word.lower()) for word in self.corpus.words() if ( word.isalpha() ) and ( len(word) > 3 ) ])
		self.all_words_dist = FreqDist(self.all_words)
		self.all_words = set([word for word in self.all_words if self.all_words_dist[word] > 2 and not( word in self.function_words ) and not( word in self.stopwords ) ])
		
		print "number of docs = ", len(self.documents_names)
		#self.extract_features()
		#self.save_features_mat("enron.mat")
		#self.save_features_numpy("features_train", "labels_train")
		#self.save_features_mat(corpus_root+".mat")
		#print self.corpus.raw(self.documents_names[165])
		#print "\n"
		#print self.features(self.documents_names[165])
		
	
	def character_based_features(self, document_name):
		document_content = self.corpus.raw(document_name)
		chars = list(document_content)
		chars_dist = FreqDist(chars)
		alphabetic_chars = [c.lower() for c in chars if(c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z') ]
		upper_case_chars = [c for c in chars if(c >= 'A' and c <= 'Z')]
		digit_chars = [c for c in chars if(c >= '0' and c <= '9')]
		special_chars = [c for c in chars if(c in self.special_chars)]
		
		num_chars = float(len(chars)) #Total number of characters(C)
		alphabetic_chars_ratio = len(alphabetic_chars)/num_chars #Total number of alphabetic characters/C
		upper_case_chars_ratio = len(upper_case_chars)/num_chars #Total number of upper-case characters/C
		digit_chars_ratio = len(digit_chars)/num_chars #Total number of digit characters/C
		white_space_ratio = chars_dist[" "]/num_chars	#Total number of white-space characters/C
		tab_space_ratio = chars_dist["\t"]/num_chars	#Total number of tab spaces/C

		#Frequency of letters (26 features)
		letter_frequency = []
		alphabetic_chars_dist = FreqDist(alphabetic_chars)
		for c in ascii_lowercase:
			letter_frequency.append(alphabetic_chars_dist[c]/num_chars)
		
		#Frequency of special characters (21 features)
		special_frequency = []
		special_chars_dist = FreqDist(special_chars)
		for c in self.special_chars:
			special_frequency.append(special_chars_dist[c]/num_chars)
		
		#prepare charactar based feature vector	
		feature_vector = []
		feature_vector.extend([num_chars, alphabetic_chars_ratio, upper_case_chars_ratio, digit_chars_ratio, white_space_ratio, tab_space_ratio])
		feature_vector.extend(letter_frequency)
		feature_vector.extend(special_frequency)
		return feature_vector
		
		
	def word_based_features(self, document_name):
		document_content = self.corpus.raw(document_name)
		words = [w.lower() for w in self.corpus.words(document_name) if w.isalpha()] #remove punctuation
		words_dist = FreqDist(words)
		sents = self.corpus.sents(document_name)
		
		num_chars = float(len(list(document_content))) #Total number of characters(C)
		num_words = float(len(words))	#Total number of words (M)
		num_unique_words = float(len(set(words))) #Number of different words
		num_sents = float(len(sents)) #Total number of sentences (S)
		
		short_words = len([w for w in words if len(w) < 4])/num_words	#Total number of short words (less than four characters)/M
		total_num_chars = sum(len(w) for w in words)/num_chars #Total number of characters in words/C
		average_word_len = sum(len(w) for w in words)/num_words	#Average word length
		
		#Average sentence length in terms of character
		average_sent_len_chars = 0
		for s in sents:
			for w in s:
				average_sent_len_chars += len(w)
		average_sent_len_chars /= num_sents
		
		average_sent_len_words = sum(len(s) for s in sents)/num_sents #Average sentence length in terms of word
		different_words_ratio = num_unique_words/num_words	#Total different words/M
		hapax_legomena = len([w for w in words if words_dist[w] == 1])	#Frequency of once-occurring words
		hapax_dislegomena = len([w for w in words if words_dist[w] == 2])	#Frequency of twice-occurring words
		
		#A vocabulary richness measure defined by Yule
		M1 = num_unique_words #M1 = tokens
		M2 = sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(words_dist.values()))]) #M2 = (V1 * 1^2) + (V2 * 2^2) + (V3 * 3^2) +
		yule_k_measure = (10000 * (M2 -M1)) / (M1 * M1)
		
		#A vocabulary richness measure defined by Simpson
		simpson_d_measure = 1 - (sum( words_dist[w] * (words_dist[w] - 1) for w in words_dist.keys()) / (num_words * (num_words - 1)))

		#A vocabulary richness measure defined by Sichele
		sichel_s_measure = hapax_dislegomena / M1
		
		#A vocabulary richness measure defined by Brune
		brunet_w_measure = 0 #TODO: implement
		
		#A vocabulary richness measure defined by Honore
		honore_r_measure = 0
		if hapax_legomena != num_unique_words:
			honore_r_measure = (100 * math.log10( num_words )) / (1 - (hapax_legomena / num_unique_words))
		
		#Frequency of words in different length (1 => 20)
		word_len_dist = FreqDist([len(w) for w in words])
		word_len_freq = [ word_len_dist[i]/num_words for i in range(1,21)]

		#prepare charactar based feature vector	
		feature_vector = []
		feature_vector.extend([num_words, short_words,total_num_chars, average_word_len, average_sent_len_chars, average_sent_len_words])
		feature_vector.extend([different_words_ratio, hapax_legomena, hapax_dislegomena, yule_k_measure, simpson_d_measure ])
		feature_vector.extend([sichel_s_measure, honore_r_measure])
		feature_vector.extend(word_len_freq)
		return feature_vector
		

	def pos_tag_name(self, tag):
		if tag in self.pos_tags:
			return tag
		return tag[0:2]
		
		
	def syntactic_based_features(self, document_name):
		words = [w.lower() for w in self.corpus.words(document_name)] #lower case
		words_dist = FreqDist(words)
		
		punctuation_freq = [words_dist[punc] for punc in self.punctuations] #Frequency of punctuations
		function_words_freq =[words_dist[w] for w in self.function_words] #Frequency of function words
		
		#Frequency of Part Of Speech 
		tags = [self.pos_tag_name(tag)for (word, tag) in pos_tag(self.corpus.words(document_name))]
		tags_dist = FreqDist(tags)
		pos_freq = [tags_dist[tag] for tag in self.pos_tags]
		
		return pos_freq + punctuation_freq + function_words_freq


	def structural_based_features(self, document_name):
		document_content = self.corpus.raw(document_name)
		paras = self.corpus.paras(document_name)
		words = self.corpus.words(document_name)
		
		num_lines = len([m.start() for m in re.finditer('\n', document_content)]) + 1	#Total number of lines
		num_sents = len(self.corpus.sents(document_name))	#Total number of sentences
		num_paras = float(len(paras))	#Total number of paragraphs
		sents_per_para = num_sents/num_paras	#Number of sentences per paragraph
		chars_per_para = len(list(document_content))/num_paras #Number of characters per paragraph
		words_per_para = len(words)/num_paras #Number of words per paragraph

		#prepare charactar based feature vector	
		feature_vector = []
		feature_vector.extend([num_lines, num_sents, num_paras, sents_per_para, chars_per_para, words_per_para])
		return feature_vector
	

	def content_based_features(self, document_name):
		words = [self.stemmer.stem(w.lower()) for w in self.corpus.words(document_name)] #lower case and stem
		words_dist = FreqDist(words)
		feature_vector = [words_dist[w] for w in self.all_words]
		return feature_vector
		
	
	# calculate a certain n-gram feature vector like uni-gram, bi-gram, etc...
	# feature_type can be frequency, binary
	def character_n_gram_features(self, document_chars, n=1, most_frequent=1000,feature_type="frequency"):
		chars_ngram = ngrams(document_chars, n)
		chars_ngram_dist = FreqDist(chars_ngram)
		if feature_type == "binary":
			feature_vector = [1 if chars_ngram_dist[ngram] > 0 else 0 for ngram in self.ngrams[n]]
		elif feature_type == "frequency":
			feature_vector = [chars_ngram_dist[ngram] for ngram in self.ngrams[n][:most_frequent]]
		return feature_vector
		
	def character_n_gram_based_features(self, document_name):
		document_content = self.corpus.raw(document_name)
		document_chars = [char for char in document_content]
		feature_vector = []
		for n in range(self.NGRAM_MIN, self.NGRAM_MAX + 1):
			feature_vector.extend(self.character_n_gram_features(document_chars, n))
		return feature_vector 
		
	def character_n_gram_variable_based_features(self, document_name):
		document_content = self.corpus.raw(document_name)
		document_chars = [char for char in document_content]
		
		chars_ngram = []
		for n in range(self.NGRAM_MIN, self.NGRAM_MAX + 1):
			chars_ngram.extend(ngrams(document_chars, n))
		chars_ngram_dist = FreqDist(chars_ngram)
		
		feature_vector = [chars_ngram_dist[ngram] for ngram in self.ngrams]
		#feature_vector = [1 if chars_ngram_dist[ngram] > 0 else 0 for ngram in self.ngrams]
		return feature_vector
	
	
	def features(self, document_name, ngram_len=1, most_frequent=1000):
		#feature_vector = self.character_based_features(document_name)
		#feature_vector.extend(self.word_based_features(document_name))
		#feature_vector.extend(self.syntactic_based_features(document_name))
		#feature_vector.extend(self.structural_based_features(document_name))
		#feature_vector.extend(self.content_based_features(document_name))
		#feature_vector = self.character_n_gram_based_features(document_name)
		#feature_vector = self.character_n_gram_variable_based_features(document_name)
		
		
		document_content = self.corpus.raw(document_name)
		document_chars = [char for char in document_content]
		feature_vector = self.character_n_gram_features(document_chars, ngram_len, most_frequent)
		return feature_vector
	
	def extract_features(self, ngram_len=1, most_frequent=1000):
		counter = 0
		self.document_features = [] # reset
		for document_name in self.documents_names:
			feature_vector =  self.features(document_name, ngram_len, most_frequent)
			#feature_vector.append(self.labels[counter]) #TODO: for weka classsification
			self.document_features.append(feature_vector)
			print counter
			counter = counter + 1
		print "total number of features = ", len(self.document_features[0])
		
	
	def select_features(self, features, labels, k=5000):
		chi = SelectKBest(chi2, k=k)
		selected_features = chi.fit_transform(features, labels)
		print "features size after selection "+ str(selected_features.shape)
		return selected_features
				
	def csv_header(self):
		#character based
		character_based_header = ["num_chars", "alphabetic_chars_ratio", "upper_case_chars_ratio", "digit_chars_ratio", "white_space_ratio", "tab_space_ratio"]
		character_based_header.extend(["char( "+c+" )" for c in ascii_lowercase])
		character_based_header.extend(["char( "+str(ord(c))+" )" for c in self.special_chars])

		#word based
		word_based_header = ["num_words", "short_words", "total_num_chars", "average_word_len", "average_sent_len_chars", "average_sent_len_words", "different_words_ratio", "hapax_legomena", "hapax_dislegomena", "yule_k_measure", "simpson_d_measure", "sichel_s_measure", "honore_r_measure"]
		word_based_header.extend(["word with len "+str(i) for i in range(1,21)])

		#syntactic based
		syntactic_based_header = ["POS( "+tag+" )" for tag in self.pos_tags]
		syntactic_based_header.extend(["PUNC( "+str(ord(punc))+" )" for punc in self.punctuations])
		syntactic_based_header.extend([w for w in self.function_words])

		#structural based
		structural_based_header = ["num_lines", "num_sents", "num_paras", "sents_per_para", "chars_per_para", "words_per_para"]

		#content based
		content_based_header = [w for w in self.all_words]

		return character_based_header + word_based_header + syntactic_based_header + structural_based_header + content_based_header + ["author_class"]

	def save_features(self, file_name = "features.csv"):
		resultFile = open(file_name,'wb')
		header = self.csv_header() #for weka
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerow(header)
		wr.writerows(self.document_features)
		
	def save_features_sample(self, file_name, authors_sample):
		sample_features = []
		
		for i in range(0,len(self.document_features)):
			if self.labels[i] in authors_sample: #include this document in the sample
				sample_features.append(self.document_features[i])
				
		resultFile = open(file_name,'wb')
		header = self.csv_header() #for weka
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerow(header)
		wr.writerows(sample_features)
		
	def from_string_labels_to_int(self):
		maxLabel = 0
		labelMap = {}
		labels = []
		for label in self.labels:
			if not label in labelMap:
				labelMap[label] = maxLabel
				maxLabel = maxLabel + 1
			labels.append(labelMap[label])
		return labels
	
	def save_features_mat(self, file_name = "features.mat"):
		scipy.io.savemat(file_name, mdict={'x': self.document_features, 'y': self.from_string_labels_to_int()})
		
	def save_features_numpy(self, x_file_name="features", y_file_name="labels"):
		numpy_features = np.asarray(self.document_features, np.float64)
		y = numpy.asarray(self.from_string_labels_to_int(), np.int32)
		#x = self.select_features(numpy_features, y)
		x = numpy_features # TODO: swap with above in variable case
		x = self.normalize_features(x)
		
		#saving to features and labels
		np.save(x_file_name, x)
		np.save(y_file_name, y)
	
	def save_features_mat_sample(self, file_name, authors_sample):
		numbered_labels = self.from_string_labels_to_int() #convert lables to int for matlab
		sample_features = []
		sample_labels = []
		for i in range(0,len(self.document_features)):
			if self.labels[i] in authors_sample: #include this document in the sample
				sample_features.append(self.document_features[i])
				sample_labels.append(numbered_labels[i])
		scipy.io.savemat(file_name, mdict={'x': sample_features, 'y': sample_labels})	
		
	
	def normalize(self, feature_vector):
		vmag = math.sqrt(sum(feature_vector[i]*feature_vector[i] for i in range(len(feature_vector))))
		return [ feature_vector[i]/vmag  for i in range(len(feature_vector)) ]
		
	# rescaling the feature vector values to be between low and high values
	# ref: http://en.wikipedia.org/wiki/Feature_scaling
	def normalize_features(self, rawpoints, high=1.0, low=0.0, axis=0, eps=1e-8):
		mins = np.min(rawpoints, axis=axis)
		maxs = np.max(rawpoints, axis=axis)
		rng = maxs - mins + eps
		return high - (((high - low) * (maxs - rawpoints)) / rng)

		
def main():
	author = AuthorIdentification("amt")
	for ngram in range(author.NGRAM_MIN, author.NGRAM_MAX + 1):
		done = False
		for l in [1000, 2000, 3000, 4000, 5000]:
			# reached max size
			if done:
				break
				
			if len(author.ngrams[ngram]) < l:
				l = len(author.ngrams[ngram]) # take all features
				done = True
				
			author.extract_features(ngram, l)
			author.save_features_numpy("out/features_"+str(ngram)+"_"+str(l), "out/labels_"+str(ngram)+"_"+str(l))
		
	#author.save_features_numpy("output/features", "output/labels")
	"""
	author_labels = list(set(author.labels))
	sample_sizes = range(5,45,5)
	number_of_samples = 5
	for sample_size in sample_sizes:
		for	i in range(1,number_of_samples + 1):
			authors_sample = random.sample(author_labels, sample_size)
			sample_corpus_path = "samples_full_csv/"+str(sample_size)+"_authors_"+str(i)+".csv"
			author.save_features_sample(sample_corpus_path, authors_sample)
			print authors_sample
	"""
	#author.save_features("f1+f2+f3.csv")
			

main()