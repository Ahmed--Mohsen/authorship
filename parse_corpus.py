#coding: utf-8

import itertools, os, glob, random, shutil, errno
from bs4 import BeautifulSoup
import csv

ofile = open("corpus.csv", 'wb')
writer = csv.writer(ofile, delimiter='\t')

file = open('weka/pan11/smallTrain.xml','r')


def readCorpusFromFile(filePath):
    f = open(filePath, 'r')
    soup = BeautifulSoup(f, "xml")
    texts = soup.find_all("text")
    for text in texts:
        #extract info
        file_name = text.get("file")
        file_name = file_name[file_name.index("/")+2:file_name.index(".")]
        author = text.author.get("id")
				#body = " ".join(text.body.string.encode('utf8').replace("\t"," ").split()).strip()
        body = (text.body.string.encode('utf8')).strip()
        if len(body.split()) < 50:
					continue
    
        #lets create a folder for this author if not exist
        label_directory = "pan_parsed_corpus/"+author
        if not os.path.exists(label_directory):
            os.makedirs(label_directory)
            
        #write document
        doc = open(label_directory+"/"+file_name+".txt", 'w+')
        doc.write(body)
        doc.close()
        
    print len(texts)
    

def readCorpusFromFolder(folderPath):
	for inFile in glob.glob( os.path.join(folderPath, '*') ):
		if os.path.isdir(inFile):
			print "Reading Folder..."+inFile
			self.readCorpusFromFolder(inFile)
		else:
			print "Reading File..."+inFile
			currentFile = open(inFile,'r')	#read current file
			label = currentFile.name.split("/")[1].split(".")[0]
			docs = currentFile.read()	
			parseDoc(docs, label)

def parseDoc(docs, label):
    #parse all docs file
    soup = BeautifulSoup(docs)
    posts = soup.find_all("post")
    
    #lets create a folder for this label if not exist
    label_directory = "parsed_corpus/"+label
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)
    
    #add related docs in the label folder
    document_counter = 0
    for post in posts:
        try:
            content = post.string.encode('utf8').replace("urlLink","").strip()
            if len(content) < 100:
              continue
        except Exception:
            continue
        document_counter += 1
        f = open(label_directory+"/"+str(document_counter)+".txt", 'w+')
        f.write(content)
        f.close()

def sampleCorpus(corpusPath):
	#read authors folders
	authors_folders = os.listdir(corpusPath)
	print authors_folders
	
	#create sampled corpus folder
	os.makedirs("corpus_samples")
	
	sample_sizes = range(5,45,5)
	number_of_samples = 10
	for sample_size in sample_sizes:
		for	i in range(1,number_of_samples + 1):
			random_sample = random.sample(authors_folders, sample_size)	
			os.makedirs("corpus_samples/"+str(sample_size)+"_authors_"+str(i))
			print random_sample	
			for	sample in random_sample:
				shutil.copytree(corpusPath+"/"+sample, "corpus_samples/"+str(sample_size)+"_authors_"+str(i)+"/"+sample)
				#os.makedirs("corpus_samples/"+str(sample_size)+"_authors_"+str(i)+"/"+sample)
#readCorpusFromFolder("corpus")
#readCorpusFromFile("smallTrain.xml")
sampleCorpus("amt")
#shutil.copytree("amt/u","test")


"""
#row = [content, label]
#writer.writerow(row)
#f.write("%s*%s \n"%(content, label))

f = open('weka/pan11/smallTrain.xml')
soup = BeautifulSoup(f)

#print(soup.prettify())
print len(soup.training.text)
documents = soup.training.text
print len(documents)
#print len(documents)
for doc in documents:
    print doc.get("file")


import xml.etree.ElementTree as ET
tree = ET.parse('weka/pan11/smallTrain.xml')
root = tree.getroot()

from xml.dom import minidom
xmldoc = minidom.parse('weka/pan11/smallTrain.xml')

itemlist = xmldoc.getElementsByTagName('item') 
for s in itemlist :
    print s.childNodes
"""