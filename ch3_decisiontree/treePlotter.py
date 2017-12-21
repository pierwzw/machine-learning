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
    createPlot.axl.annotate(nodeTxt, xy=parentpt, xycoords='axes fraction',\
                    xytext=centerPt, textcoords='axes fraction',\
                    va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

createPlot()

