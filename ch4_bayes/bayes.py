from numpy import *
import re
import operator

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
    p0Denom, p1Denom = 2.0, 2.0#修正这2.0之前为0.0
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


def bagOfWord2VecMN(vocabList, inputSet):
    """
    朴素贝叶斯词袋模型
    :param vocalList:
    :param inputSet:
    :return:
    """

    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textparse(bigString):
    """
    文件解析
    :param bigString:
    :return:
    """
    #不能为\W*会导致空匹配异常
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    """
    垃圾邮件测试函数
    注：此法为留出法，需多次计算取平均，若要更精确，可以使用k折
    :return:
    """

    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        wordList = textparse(open('email\\spam\\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        #1为垃圾邮件， 0为非垃圾邮件
        classList.append(1)
        #有不可识别的符号
        wordList = textparse(open('email\\ham\\%d.txt' % i, encoding='gb18030', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    #vocabList = list(set(fullText))
    #python3中range不返回list
    trainingSet, testSet = list(range(50)), []
    #随机构建训练集(而不是测试集)
    for i in range(10):
        #uniform生成(m,n)之间的实数
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        #trainMat为每一篇文章中词的个数
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    return float(errorCount)/len(testSet)


def calcMostFreg(vocabList, fullText):
    """
    高频词去除函数
    :param vocabList:
    :param fullText:
    :return:
    """

    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def removeStopedWords(vocabList):
    """
    去除停用词函数
    :param vocabList:
    :return:
    """

    stopedWordsList = open('stopword.txt', 'r').readlines()
    print(len(vocabList))
    stopedWordsList = [word.strip() for word in stopedWordsList]
    for word in vocabList:
        if word in stopedWordsList:
            vocabList.remove(word)
    print(len(vocabList))
    return vocabList


def localWords(feed1, feed0):
    """
    RSS源分类器
    此处错误率为占所有用词的百分比
    :param feed1:
    :param feed0:
    :return:
    """

    import feedparser
    docList, classList, fullText = [], [], []
    minLen = min(len(feed1.entries), len(feed0.entries))
    for i in range(minLen):
        wordList = textparse(feed1.entries[i].summary)
        docList.append((wordList))
        fullText.extend(wordList)
        classList.append(1)
        wordList = textparse(feed0.entries[i].summary)
        docList.append((wordList))
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreg(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    #去除停用词
    vocabList = removeStopedWords(vocabList)
    trainingSet, testSet = list(range(2*minLen)), []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        #trainMat为每一篇文章中词的个数
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is : ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V, float(errorCount)/len(testSet)


def getTopWords(ny, sf):
    """
    最具表征性的词汇显示函数
    :param ny:
    :param sf:
    :return:
    """

    vocabList, p0V, p1V, errorRate = localWords(ny, sf)
    topNY, topSF = [], []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])






















