import csv

data = {}
with open("deep.csv") as f:
	lis=[line.split() for line in f] 
	for i,x in enumerate(lis):
		exp, size, noise, pre, fine, accuracy, tr, vl = x[0].split(",")
		accuracy = float(accuracy); size = int(size); exp = "_".join(exp.split("_")[:-1])
		#print exp, size, accuracy
		
		if not exp in data:
			data[exp] = [0] * 10
		index = size/1000 - 1
		data[exp][index] = max(data[exp][index], accuracy)

# save results
file = open('deep-results-3.csv','w')

# header
header = ["size"] + [str(i) for i in range(1000, 10001, 1000)]
file.write(",".join(header)+ "\n")

for key in data:
	#print key
	row = [key] + [str(x) for x in data[key]]
	#print row
	file.write(",".join(row)+ "\n")

file.close()