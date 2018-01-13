from math import *
from numpy import *
import random
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat, lableMat = [], []
    fr = open('dataset\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 1.0是为了匹配常量w0,即w0*1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        lableMat.append(int(lineArr[2]))
    return dataMat, lableMat


def sigmoid(inX):
    """
    阶跃函数
    :param inX:
    :return:
    """

    return float(1.0/(1 + exp(-inX)))


def gradAscent(dataMatIn, classLables):
    """
    BGA/BGD
    若为梯度下降则error和alpha的符号要同时改变，实际结果相同
    Logistic回归批量梯度上升优化算法(采用sigmoid为性能度量)
    dataMatIn为二维numpy数组， 行为样本，列为特征
    此处也可以使用最大化对数似然（周:p.59）
    或者均方误差(周:p.54)，均方误差为回归任务中最常用的性能度量
    :param dataMatIn:
    :param classLables:
    :return:
    """

    dataMatrix = mat(dataMatIn)
    # 求转置
    lableMat = mat(classLables).T
    m, n = shape(dataMatrix)
    # m, n = dataMatrix.shape
    #alpha为步长
    alpha = 0.001
    maxCycles = 100
    # weight即为回归系数初始化为1,为nx1,n为特征数
    weight = ones((n, 1))
    for k in range(maxCycles):
        # 此处矩阵相乘已经包括了求和
        h = sigmoid(dataMatrix * weight)
        error = lableMat - h
        weight += alpha * dataMatrix.T * error
        # weight
    return weight


def stocGradAscen1(dataMatrix, classLables):
    """
    SGA/SGD
    若为梯度下降则error和alpha的符号要同时改变，实际结果相同
    Logistic回归随机梯度上升优化算法(采用sigmoid为性能度量)
    dataMatIn为二维numpy数组， 行为样本，列为特征
    此处也可以使用最大化对数似然（周:p.59）
    或者均方误差(周:p.54)，均方误差为回归任务中最常用的性能度量
    :param dataMatIn:
    :param classLables:
    :return:
    """

    m, n = shape(dataMatrix)
    dataMatrix = mat(dataMatrix)
    #alpha为步长
    alpha = 0.01
    # weight即为回归系数初始化为1,为nx1,n为特征数
    weight = mat(ones(n))
    maxCycles = 500
    for i in range(maxCycles):
        randIndex = int(random.uniform(0, m))
        h = sigmoid(dataMatrix[randIndex] * weight.T)
        error = classLables[randIndex] - h
        weight += alpha * error * dataMatrix[randIndex]
    return weight.tolist()[0]


def stocGradAscen2(dataMatrix, classLables):
    """
    Logistic回归随机梯度上升优化算法(采用sigmoid为性能度量)
    :param dataMatIn:
    :param classLables:
    :return:
    """

    m, n = shape(dataMatrix)
    #alpha为步长
    alpha = 0.01
    # weight即为回归系数初始化为1,为nx1,n为特征数
    weight = ones(n)
    maxCycles = 1000
    for i in range(maxCycles):
        randIndex = int(random.uniform(0, m))
        h = sigmoid(sum(dataMatrix[randIndex] * weight))
        error = classLables[randIndex] - h
        # 不要忘了加array
        weight += array(alpha * error) * dataMatrix[randIndex]
    return weight


def stocGradAscen3(dataMatrix, classLabels, numIter=300):
    """
    每次取的index都不一样即在整个数据集上都运行了一次，精度将大大提高
    结果显示只分错了五个点
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            # 此处达到正负20000时会报float溢出， 可使用longfloat
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            # 不要忘了加array
            weights = weights + array(alpha * error) * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def leastSquare(xArr, yArr):
    """
    最小二乘法(确定性算法不是优化算法)
    X必须为列满秩
    :param dataMatrix:
    :param classLables:
    :return:
    :see http://blog.csdn.net/sinat_16233463/article/details/37363183
    """

    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * (xMat.T * yMat)
    # return [k[0] for k in ws.tolist()]
    # getA()为返回array
    return ws.getA()



def plotBestFit(dataMat, lableMat, weights):
    """
    画出数据集和logistic逻辑回归最佳拟合的函数
    :param wei:
    :return:
    """

    dataArr = array(dataMat)
    m, n = shape(dataArr)
    x1, x2, y1, y2 = [], [], [], []
    for i in range(m):
        if int(lableMat[i] == 1):
            x1.append(dataArr[i, 1])
            y1.append(dataArr[i, 2])
        else:
            x2.append(dataArr[i, 1])
            y2.append(dataArr[i, 2])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')
    # arange()类似于内置函数range()，通过指定开始值、终值和步长创建表示等差数列的一维数组，注意得到的结果数组不包含终值。
    # linspace()通过指定开始值、终值和元素个数创建表示等差数列的一维数组，可以通过endpoint参数指定是否包含终值，默认值为True，即包含终值。
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    """
    Logistic回归分类函数
    :param inX:
    :param weights:
    :return:
    """

    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    运用随机梯度上升算法来预测疝病马的死亡率
    :return: 
    """

    frTrain = open('dataset\\horseColicTraining.txt')
    frTest = open('dataset\\horseColicTest.txt')
    trainingSet, trainingLables = [], []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = [float(currLine[i]) for i in range(21)]
        trainingSet.append(lineArr)
        trainingLables.append(float(currLine[21]))
    trainWeights = stocGradAscen3(array(trainingSet), trainingLables, 500)
    errorCount, numTestVec = 0, 0.0
    lines = frTest.readlines()
    numTestVec = len(lines)
    for line in lines:
        currLine = line.strip().split('\t')
        lineArr = [float(currLine[i]) for i in range(21)]
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print('the error rate of this test is: %f' % errorRate)
    return errorRate


def multiTest():
    """
    调用预测函数多次取平均值
    :return:
    """

    numTests, errorSum = 10, 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f' % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    multiTest()
    # dataArr, lableMat = loadDataSet()
    # weights = stocGradAscen3(dataArr, lableMat)
    # plotBestFit(dataArr, lableMat, weights)























