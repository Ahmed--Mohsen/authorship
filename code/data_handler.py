import csv
import numpy as np
import theano
from sklearn.cross_validation import train_test_split

def load_data(features_path="data/features_ngram_huge.npy", labels_path="data/labels_ngram_huge.npy"):
	# reading full dataset
	features = np.load(features_path)
	labels = np.load(labels_path)
	
	# splitting data
	train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(features, labels, test_size=0.3, random_state=42)
	
	test_set_x = theano.shared(value=test_set_x, name='test_set_x', borrow=True)
	test_set_y = theano.shared(value=np.array(test_set_y), name='test_set_y', borrow=True)
	
	# split train set into validation set
	train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(train_set_x, train_set_y, test_size=0.25, random_state=13)
	
	print train_set_x.shape, valid_set_x.shape, test_set_x.get_value(borrow=True).shape
	
	train_set_x = theano.shared(value=train_set_x, name='train_set_x', borrow=True)
	train_set_y = theano.shared(value=np.array(train_set_y), name='train_set_y', borrow=True)
	
	valid_set_x = theano.shared(value=valid_set_x, name='valid_set_x', borrow=True)
	valid_set_y = theano.shared(value=np.array(valid_set_y), name='valid_set_y', borrow=True)
	
	return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))
	

def load_dataset(path_id="", folder="", use_float_32=False, test_ratio=0.3, valid_ratio=0.1):	
#def load_dataset(path_id="", use_float_32=False, test_ratio=0.2, valid_ratio=0.1):
	# reading full dataset
	features_path = "data-cross/%s/features%s.npy"%(folder, path_id)
	labels_path = "data-cross/%s/labels%s.npy"%(folder, path_id)
	

	features = np.load(features_path)
	if use_float_32:
		features = features.astype(np.float32)
	labels = np.load(labels_path)
	
	# splitting data
	train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(features, labels, test_size=test_ratio, random_state=89677)
	#train_set_x = features[:2500]
	#train_set_y = labels[:2500]
	
	#test_set_x = features[2500:]
	#test_set_y = labels[2500:]
	test_set_x = theano.shared(value=test_set_x, name='test_set_x', borrow=True)
	test_set_y = theano.shared(value=np.array(test_set_y), name='test_set_y', borrow=True)
	
	# split train set into validation set
	train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(train_set_x, train_set_y, test_size=valid_ratio, random_state=89677)
	
	print train_set_x.shape, valid_set_x.shape, test_set_x.get_value(borrow=True).shape
	
	train_set_x = theano.shared(value=train_set_x, name='train_set_x', borrow=True)
	train_set_y = theano.shared(value=np.array(train_set_y), name='train_set_y', borrow=True)
	
	valid_set_x = theano.shared(value=valid_set_x, name='valid_set_x', borrow=True)
	valid_set_y = theano.shared(value=np.array(valid_set_y), name='valid_set_y', borrow=True)
	
	return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))	
	
	
def load_full_dataset(path_id="", folder="", use_float_32=False):
	# reading full dataset
	features_path = "data-cross/%s/features%s.npy"%(folder, path_id)
	labels_path = "data-cross/%s/labels%s.npy"%(folder, path_id)
	domains_path = "data-cross/%s/labels%s-domain.csv"%(folder, path_id)
	
	features = np.load(features_path)
	if use_float_32:
		features = features.astype(np.float32)
	#labels = np.load(labels_path)
	
	x = theano.shared(value=features, name='features', borrow=True)
	y = open(domains_path,'r').read().splitlines()
	#y = theano.shared(value=labels, name='labels', borrow=True)
	
	return (x, y)	
	
	
def load_reuters_dataset(valid_ratio=0.1, path_id=""):
	train_features_path = "data/features_train%s.npy"%(path_id)
	train_labels_path = "data/labels_train%s.npy"%(path_id)
	
	test_features_path = "data/features_test%s.npy"%(path_id)
	test_labels_path = "data/labels_test%s.npy"%(path_id)

	# reading test dataset
	test_set_x = np.load(test_features_path) #.astype(np.float32)
	test_set_y = np.load(test_labels_path)
	test_set_x = theano.shared(value=test_set_x, name='test_set_x', borrow=True)
	test_set_y = theano.shared(value=np.array(test_set_y), name='test_set_y', borrow=True)
	
	# reading train dataset
	train_set_x = np.load(train_features_path) #.astype(np.float32)
	train_set_y = np.load(train_labels_path)
	
	# splitting training dataset into train and validation (80% 20%)
	train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(train_set_x, train_set_y, test_size=valid_ratio, random_state=42)
	
	train_set_x = theano.shared(value=train_set_x, name='train_set_x', borrow=True)
	train_set_y = theano.shared(value=np.array(train_set_y), name='train_set_y', borrow=True)

	valid_set_x = theano.shared(value=valid_set_x, name='valid_set_x', borrow=True)
	valid_set_y = theano.shared(value=np.array(valid_set_y), name='valid_set_y', borrow=True)

	print test_set_x.get_value(borrow=True).shape, valid_set_x.get_value(borrow=True).shape, test_set_x.get_value(borrow=True).shape

	return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))

def load_full_data(features_path="data/features_ngram_huge.npy", labels_path="data/labels_ngram_huge.npy"):
	# reading full dataset
	features = np.load(features_path)
	labels = np.load(labels_path)
	
	x = theano.shared(value=features, name='features', borrow=True)
	y = theano.shared(value=labels, name='labels', borrow=True)
	return (x, y)
	
	
def save_data(features, labels, file_name = "features.csv"):
	print labels
	n = len(features[0])
	resultFile = open(file_name,'wb')
	wr = csv.writer(resultFile, dialect='excel')
	
	# write header
	header =  ["feature_"+str(i) for i in range(n)] + ["class"] 
	wr.writerow(header)
	
	# write the feature vectors appended to there class
	feature_vectors = features.tolist()
	for i in range(len(features)):
		feature_vectors[i].append("author_" + str(int(labels[i])))
	wr.writerows(feature_vectors)
	
	
def save_features(features, labels, file_name = "features.csv"):
	print labels
	n = len(features[0])
	resultFile = open(file_name,'wb')
	wr = csv.writer(resultFile, dialect='excel')
	
	# write header
	header =  ["feature_"+str(i) for i in range(n)] + ["class"] 
	wr.writerow(header)
	
	# write the feature vectors appended to there class
	feature_vectors = features.tolist()
	for i in range(len(features)):
		feature_vectors[i].append("author_" + str(int(labels[i])))
	wr.writerows(feature_vectors)
	
	
def split_data(file_name="features_reuters.csv"):
	import csv
	with open(file_name, 'rb') as f:
		reader = csv.reader(f)
		features = list(reader)
		header = features[0]
		

	resultFileTrain = open("features_reuters_train.csv",'wb')
	wr = csv.writer(resultFileTrain, dialect='excel')
	wr.writerows(features[0:2501])
	
	resultFileTest = open("features_reuters_test.csv",'wb')
	wr = csv.writer(resultFileTest, dialect='excel')
	wr.writerow(header)
	wr.writerows(features[2501:5001])
	
	


###############################
""""
datasets = load_dataset("_all")
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

train_set_x = train_set_x.eval()
valid_set_x = valid_set_x.eval()
test_set_x = test_set_x.eval()

train_set_y = train_set_y.get_value(borrow=True)
valid_set_y = valid_set_y.get_value(borrow=True)
test_set_y = test_set_y.get_value(borrow=True)

x = np.concatenate((train_set_x, valid_set_x))
x = np.concatenate((x, test_set_x))
print x.shape
print x

y = np.concatenate((train_set_y, valid_set_y))
y = np.concatenate((y, test_set_y))
print y.shape
print y
"""
