from math import *
from numpy import *
import random


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

    return 1.0/(1 + exp(-inX))


def gradAscent(dataMatIn, classLables):
    """
    BGA/BGD
    若为梯度下降则error和alpha的符号要同时改变，实际结果相同
    Logistic回归批量梯度上升优化算法(采用sigmoid为性能度量)
    dataMatIn为二维numpy数组， 行为样本，列为特征
    此处为以阶跃函数与标签差作为梯度，精度较小，也可以使用最大化对数似然（周:p.59）
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


def gradAscent2(dataMatrix, classLables):
    """
    SGA/SGD
    若为梯度下降则error和alpha的符号要同时改变，实际结果相同
    Logistic回归随机梯度上升优化算法(采用sigmoid为性能度量)
    dataMatIn为二维numpy数组， 行为样本，列为特征
    此处为以阶跃函数与标签差作为梯度，精度较小，也可以使用最大化对数似然（周:p.59）
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


def gradAscent3(dataMatrix, classLables):
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
    maxCycles = 500
    for i in range(maxCycles):
        randIndex = int(random.uniform(0, m))
        h = sigmoid(sum(dataMatrix[randIndex] * weight))
        error = classLables[randIndex] - h
        weight += array(alpha * error) * dataMatrix[randIndex]
    return weight


def leastSquare(xArr, yArr):
    """
    最小二乘法(确定性算法不是优化算法)
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
    return [k[0] for k in ws.tolist()]





















