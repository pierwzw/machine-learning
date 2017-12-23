from numpy import *


def loadDataSet():
    """
    创建文档实验样本
    :return:postingList：词条切分后的文档集合
            classVec：   类别标签
    """

    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建包含在文档中出现的所有不重复词的列表
    :param dataSet:
    :return:
    """

    vocabSet = set()
    for doc in dataSet:
        vocabSet |= set(doc)  # |为集合并
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型：一个词在文档中只出现一次
    输出文档向量对应着词汇表中的词是否在文档中出现
    :param vocabList:
    :param inputSet:
    :return:
    """

    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word ', word, ' is not in my Vocabulary!')
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    """
    词袋模型：一个词在文档可出现多次
    输出文档向量对应着词汇表中的词是否在文档中出现
    :param vocabList:
    :param inputSet:
    :return:
    """

    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix:
    :param trainCategory:
    :return:
    """

    # trainMatrix的维数为所有文档中不重复词的数量
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 此处不修正，但应该修正
    pAbusvie = sum(trainCategory) / float(numTrainDocs)
    # piNum为所有i类文档中所有词出现的数量和向量
    # piDenom为所有i类文档中所有词的数量和（词只出现一次）
    # 即sum(piNum) = piDenom

    # 若出现概率为0的情况则应使用拉普拉斯修正,此例中不会出现，若词典词数大于训练词
    # 数则可能会出现，而此例中词典词数和训练词数相同
    #p0Num, p1Num = zeros(numWords), zeros(numWords)
    p0Num, p1Num = ones(numWords), ones(numWords)
    p0Denom, p1Denom = 0.0, 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 若出现下溢出则使用对数化
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    #p1Vect = p1Num / p1Denom
    #p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusvie


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """

    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB(testDoc):
    """
    朴素贝叶斯训练函数
    :param testDoc:
    :return:
    """

    listOPosts, listClasses = loadDataSet()
    myVoacbList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVoacbList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # 必须使用array，否则不能使用内积
    docVec = array(setOfWords2Vec(myVoacbList, testDoc))
    return classifyNB(docVec, p0V, p1V, pAb)
