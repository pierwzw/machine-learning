from ch3_decisiontree import trees, treePlotter

# fr = open('contact_lenses\\lenses.txt')
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lensesLables = ['age', 'prescript', 'astigmatic', 'tearRate']
# lensesTree = trees.createTree(lenses, lensesLables)
# #print(lensesTree)
# treePlotter.createPlot(lensesTree)

dataset = [['di','bei', 'bo','di2'],
           ['gao','bei','ben','zhong2'],
           ['di', 'nan', 'ben','gao2'],
           ['gao','bei','yan', 'zhong2']]
print(trees.chooseBestFeatureToSplit(dataset))
