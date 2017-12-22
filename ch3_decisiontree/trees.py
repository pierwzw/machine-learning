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
    """
    创建鱼类数据集
    :return:
    """

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
    ID3决策树划分方式：香农信息增益
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """

    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        #从列表中创建集合是Python语言得到列表中唯一元素值的最快方法
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy+= prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i#第i个feature
    return bestFeature


def majorityCnt(classList):
    """
    投票法决定叶子节点的分类
    :param classList:
    :return:
    """

    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #itemgetter:取该类的样本数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, lables):
    """
    创建ID3决策树
    :param dataSet:
    :param lables:
    :return:
    """

    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = lables[bestFeat]
    myTree = {bestFeatLable:{}}
    del(lables[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #list为引用传递，为防止生命周期中更改
        subLables = lables[:]
        myTree[bestFeatLable][value] = createTree(splitDataSet\
            (dataSet, bestFeat, value), subLables)
    return myTree


def classify(inputTree, featLables, testVec):
    """
    使用决策树的分类函数
    :param inputTree:
    :param featLables:
    :param testVec:
    :return:
    """

    #没有属性可以匹配时返回默认的分类：no
    classLable = 'no'
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLable = classify(secondDict[key], featLables, testVec)
            else:
                classLable = secondDict[key]
    return classLable


def storeTree(inputTree, filename):
    """
    序列化树到磁盘，因为创建决策树很耗时，即使是很小的数据集
    :param inputTree:
    :param filename:
    :return:
    """

    import pickle
    #2中用'w',3中用'wb'
    # a+ 可读写模式，写只能写在文件末尾
    #w+ 可读写，与a+的区别是要清空文件内容
    #r+ 可读写，与a+的区别是可以写到文件任何位置
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
    从磁盘中反序列化决策树
    :param filename:
    :return:
    """

    import pickle
    fr = open(filename,'rb')#yeah, work on!
    #fr = open(filename, 'r', encoding='UTF-8'):Oops! can't work!
    return pickle.load(fr)

















