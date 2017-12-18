from numpy import *
import matplotlib.pyplot as plt
from ch2_knn import kNN

datingDataMat, datingLables = kNN.file2matrix('datedata\datingTestSet2.txt')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

print(normMat, "\n", ranges, '\n', minVals)