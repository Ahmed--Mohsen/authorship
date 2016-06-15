import numpy as np
#
#x = np.load("out/features_3_2000.npy")
#print x.shape
#print x


"""
y = np.load("output/labels.npy")
print y.shape

y = list(y)
for a in range(50):
	print y[:2500].count(a)
	print y[2500:].count(a)
"""

import glob
#print glob.glob("data/label*.npy")
labels = glob.glob("data/*/features*.npy")
for label in labels:
	y = np.load(label)
	print y.shape
	print y
