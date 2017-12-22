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
        vocabSet |= set(doc)#|为集合并
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    输出文档向量对应着词汇表中的词是否在文档中出现
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word ', word, ' is not in my Vocabulary!')
    return returnVec




