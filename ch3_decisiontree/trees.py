from functools import reduce
from math import log
from collections import Counter
import operator


def calcShannonEnt(dataSet):
    """
    计算香农熵
    :param dataSet:
    :return:
    """

    numEntries = len(dataSet)
    lableCounts = {}
    for featVec in dataSet:#取mat的每一行？不用dataSet[i,:]?
        currentLable = featVec[-1]
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] += 1
    shannonEnt = 0.0
    for key in lableCounts:
        prob = float(lableCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def calcShannonEnt2(dataSet):
    """
    计算香农熵
    :param dataSet:
    :return:
    """

    classCount = Counter(sample[-1] for sample in dataSet)
    prob = [float(v) / sum(classCount.values()) for v in classCount.values()]
    return reduce(operator.add, map(lambda x: -x * log(x, 2), prob))


def calcGiniIndex(dataSet):
    """
    计算基尼指数, 基尼值越小，纯度越高
    :param dataSet:
    :return:
    """

    numEntries = len(dataSet)
    lableCounts = {}
    for featVec in dataSet:  # 取mat的每一行？不用dataSet[i,:]?
        currentLable = featVec[-1]
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] += 1
    giniIndex = 1.0
    for k in lableCounts:
        prob = float(lableCounts[k])/numEntries
        giniIndex -= prob**2
    return giniIndex


def calcGiniIndex2(dataSet):
    """
    计算基尼指数，  基尼值越小，纯度越高
    :param dataSet:
    :return:
    """

    labelCounts = Counter(sample[-1] for sample in dataSet)
    prob = [float(v) / sum(labelCounts.values()) for v in labelCounts.values()]
    return 1 - reduce(operator.add, map(lambda x: x ** 2, prob))



def createDateSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    lables = ['no surfacing', 'flippers']
    return dataSet, lables


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet:
    :param axis:
    :param value:
    :return:
    """
    retDataSet = []#函数内部修改会影响整个生命周期
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """

    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy+= prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy = newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i#第i个feature
    return bestFeature










