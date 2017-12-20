from numpy import *
import operator

from os import listdir


def createDataSet():
    """
    创建数据集
    :return:
    """

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classisy0(inX, dataSet, lables, k):
    """
    kNN分类器
    :param inX:
    :param dataSet:
    :param lables:
    :param k:
    :return:
    """

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
    """
    数据矩阵化
    :param filename:
    :return:
    """

    fr = open(filename)
    arrayOLines:list = fr.readlines()
    numberOfLines:int = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))#生成nxn的零数组
    classLableVector:list = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]#index为第几行
        classLableVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLableVector

def autoNorm(dataSet):
    """
    归一化特征矩阵
    :param dataSet:
    :return:
    """

    minVals = dataSet.min(0)#0取每列，1取每行
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    测试错误率
    :return:
    """

    hoRatio = 0.70
    datingDataTestMat, datinglables = file2matrix('datedata\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataTestMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    k = 10
    for i in range(numTestVecs):
        classifierResult = classisy0(normMat[i,:], normMat[numTestVecs:m,:],\
                                     datinglables[numTestVecs:m], k)
        print('the classify came back with: ', classifierResult, ', the real answer'\
                                                                 ' is: ', datinglables[i])
        if (classifierResult != datinglables[i]):
            errorCount += 1
    print('the total error rate is: ', errorCount/float(numTestVecs))
    print('end')

def classifyPerson():
    """
    使用分类器
    :return:
    """

    resultList = ['not at all', 'in small doses', 'in large doses']
    persentTats = float(input('persntage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per yeat?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLables = file2matrix('datedata\datingTestSet2.txt')
    normMat, ranges, minVlas = autoNorm(datingDataMat)
    inArr = array([ffMiles, persentTats, iceCream])
    classifierResult = classisy0((inArr - minVlas)/ranges, normMat, datingLables, 3)
    print('Your probably like this person: ', resultList[classifierResult - 1])

'''以下为手写识别'''
def img2vector(filename):
    """
    将一个32×32的二进制图像矩阵转换为1×1024的向量
    :param filename:
    :return:
    """

    returnVect = zeros((1, 1024))#必须要有（）,否则不能识别类型
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])#也可使用arr[][]
    return returnVect

def handwritingClassTest():
    hwLables = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits\\'+fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits\\'+fileNameStr)
        classifierResult = classisy0(vectorUnderTest, trainingMat, hwLables, 3)
        print('the classifier came back with: ', classifierResult, ', thereal answer is: ', classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1
    print('\nthe total number of errors is: ', errorCount)
    print('\nthe total error rate is : ', (errorCount/float(mTest)))


















