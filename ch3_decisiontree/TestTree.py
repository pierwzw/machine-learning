from  ch3_decisiontree import trees, treePlotter

myDat, lables = trees.createDateSet()
myTree = treePlotter.retrieveTree(0)
trees.storeTree(myTree, 'c:\\tree.txt')
mytree = trees.grabTree('c:\\tree.txt')
print(mytree)
# lable = trees.classify(myTree, lables, [1, 2])
# print(lable)
#myTree['no surfacing'][3] = 'maybe'
#treePlotter.createPlot(myTree)
# print(myTree)
# leafs = treePlotter.getNumLeafs(myTree)
# print(leafs)
# depth = treePlotter.getTreeDepth(myTree)
# print(depth)
#treePlotter.createPlot()
# myDat, lables = trees.createDateSet()
#
# myTree = trees.createTree(myDat, lables)

#print(myTree)
# bestFeatureIndex = trees.chooseBestFeatureToSplit(myDat)
#
# print(myDat,'\n',bestFeatureIndex)

# print(myDat)
#
# shannonEnt = trees.calcShannonEnt(myDat)
# print(shannonEnt)
#
# shannonEnt = trees.calcShannonEnt2(myDat)
# print(shannonEnt)
#
# giniIndex = trees.calcGiniIndex(myDat)
# print(giniIndex)
#
# giniIndex = trees.calcGiniIndex2(myDat)
# print(giniIndex)
