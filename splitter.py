import numpy
from sklearn.svm import LinearSVC

class Splitter(object):
	
	def __init__(self, features, labels):
		super(Splitter, self).__init__()
		self.features = features
		self.labels = labels
		self.topics = ["Politics", "Society", "World", "UK"]
		self.genres = [self.topics, "Books"]
		
		# hold doamin/topic of each doc
		self.domains = [label.split("/")[0] for label in self.labels]
		self.authors = [label.split("/")[-1] for label in self.labels]
	
	def split(self):
		splits = self.cross_topic()
		splits += self.out_of_topic()
		splits += self.cross_genre()
		return splits
	
	def split_cross(self, source_domain, dest_domain):
	
		# hold x and y data for source topic
		x_train_indx = []; y_train = []
	
		# hold x and y data for dest topic
		x_test_indx = []; y_test = []
	
		# loop over all corpus docs
		for i in range(len(self.labels)):
			author = self.authors[i]
			domain = self.domains[i]
		
			if domain in source_domain:
				x_train_indx.append(i)
				y_train.append(author)
		
			elif domain in dest_domain:
				x_test_indx.append(i)
				y_test.append(author)
	
		# select train and test vectors from x
		x_train = self.features[x_train_indx, :]
		x_test = self.features[x_test_indx, :]
		return (x_train, y_train, x_test, y_test)
		
	
	def classify(self, x_train, y_train, x_test, y_test):
		clf = LinearSVC()
	
		# train the model
		clf.fit(x_train, y_train)
	
		# predict labels of x_test
		predicted = clf.predict(x_test)
	
		#print (metrics.classification_report(y_test, predicted, target_names=set(y_train)))
		return numpy.mean(predicted == y_test)
		
		
	def cross_domain(self, domains):
		# hold all cross domain splits split = (domain_a, domain_b, accuracy)
		splits = []
		
		for i in range(len(domains)):
			for j in range(len(domains)):
				# skip intra topics
				if i == j:
					continue
					
				# split data according to cross topics selection
				x_train, y_train, x_test, y_test = self.split_cross(domains[i], domains[j])
				accuracy = self.classify(x_train, y_train, x_test, y_test) * 100
				splits.append( (self.domain_name(domains[i]), self.domain_name(domains[j]), accuracy) )				
		return splits
	
	def cross_topic(self):
		return self.cross_domain(self.topics)
		
	def cross_genre(self):
		return self.cross_domain(self.genres)
		
	def out_of_topic(self):
		splits = []
		for i in range(len(self.topics)):
			out_of_topic = [topic for topic in self.topics if topic != self.topics[i]]
			x_train, y_train, x_test, y_test = self.split_cross(out_of_topic, self.topics[i])
			accuracy = self.classify(x_train, y_train, x_test, y_test) * 100
			splits.append( (self.domain_name(out_of_topic), self.domain_name(self.topics[i]), accuracy) )				
		return splits

	
	def domain_name(self, domain):
		# single domain name
		if isinstance(domain, str):
			return domain
		
		# multiple domain (cross-genre)
		if domain == self.topics:
			return "Opinions"
		
		# multiple domain (out of topic)
		out_topic = "".join( set(self.topics) - set(domain) )
		return "Opinions-"+out_topic
		
		