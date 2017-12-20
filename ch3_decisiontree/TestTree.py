from  ch3_decisiontree import trees

myDat, lables = trees.createDateSet()

print(myDat)

shannonEnt = trees.calcShannonEnt(myDat)
print(shannonEnt)

shannonEnt = trees.calcShannonEnt2(myDat)
print(shannonEnt)

giniIndex = trees.calcGiniIndex(myDat)
print(giniIndex)

giniIndex = trees.calcGiniIndex2(myDat)
print(giniIndex)
