import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from ch2_knn import kNN

group, lables = kNN.createDataSet()

datingDataMat, datingLables = kNN.file2matrix('datedata\datingTestSet2.txt')

#print(datingDataMat, '\n', datingLables)

fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot(111)
plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show


