from ch4_bayes import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVoacbList = bayes.createVocabList(listOPosts)
print(myVoacbList)