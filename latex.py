def get_coor(s):
	acc = s.split()
	coor = ["(%d,%s)"%((i+1) * 1000, acc[i]) for i in range(10)]
	return "".join(coor)


print get_coor("71.9	85.48	89.4	91.02	92.66	93.2	94.22	94.3	94.78	95.12")
	