import matplotlib.pyplot as plt
import matplotlib

#定义自定义字体，文件名是系统中文字体
#myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/simhei.ttf')
# #解决负号'-'显示为方块的问题
#matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams['font.sans-serif']=['SimHei']

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentpt, nodeType):
    #xytext为节点坐标
    createPlot.axl.annotate(nodeTxt, xy=parentpt, xycoords='axes fraction',\
                    xytext=centerPt, textcoords='axes fraction',\
                    va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlotSimple():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlotSimple.axl = plt.subplot(111, frameon=False)
    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    """
    获取叶节点的数目
    :param myTree:
    :return:
    """

    numLeafs = 0
    # 如果使用的是python2 firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    获取树的层数
    :param myTree:
    :return:
    """

    maxDepth = 0
    # 3中返回的是dict_keys如果使用的是python2 firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTree = [{
                    'no surfacing': {
                        0: 'no',
                        1: {
                            'flippers': {
                                0: 'no',
                                1: 'yes'
                            }
                        }
                    }
                },
                {
                    'no surfacing': {
                        0: 'no',
                        1: {
                            'flippers': {
                                0: {
                                    'head': {
                                        0: 'no',
                                        1: 'yes'
                                    }
                                },
                                1: 'no'
                            }
                        }
                    }
                }]
    return listOfTree[i]


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    :param cntrPt:
    :param parentPt:
    :param txtString:
    :return:
    """

    xMid = (parentPt[0] + cntrPt[0])/2
    yMid = (parentPt[1] + cntrPt[1])/2
    createPlot.axl.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制树形图
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """

    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff -= 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff += 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff += 1.0/plotTree.totalD


def createPlot(inTree):
    """
    绘制树形图主函数
    :param inTree:
    :return:
    """

    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    #全局变量存储树的宽度和深度
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


