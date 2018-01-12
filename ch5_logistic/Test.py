from ch5_logistic import logRegres as lr


dataArr, lableMat = lr.loadDataSet()
ws = lr.leastSquare(dataArr, lableMat)
print(ws)