from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classisy0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]#二维数组的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet#tile([1,2],(2,3)) 重复[1, 2]在行上2次，在列上3次
    sqDiffMat = diffMat**2#直接对矩阵里的元素求平方
    sqDistances = sqDiffMat.sum(axis=1)#axis为0对列求和，为1对行求和
    distances = sqDistances**0.5
    sortedDistancies = distances.argsort()#返回数组值从小到大的索引值，以排序后为基,即原数组的下标排序
    classCount = {}
    for i in range(k):
        voteIlable = lables[sortedDistancies[i]]#sortedDistancies[i]为distances排序
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1#不存在则返回0
    #sort()与sorted()的不同在于，sort是在原位重新排列列表，而sorted()是产生一个新的列表。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]#返回最大的种类数

def file2matrix(filename):
    fr = open(filename)
    arrayOLines:list = fr.readlines()
    numberOfLines:int = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLableVector:list = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]#index为第几行
        classLableVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLableVector









