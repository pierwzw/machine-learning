from ch4_bayes import bayes

testDoc1 = ['love', 'my', 'dalmation']
testDoc2 = ['dog', 'my', 'love']
classLable = bayes.testingNB(testDoc1)
print(classLable)
# listOPosts, listClasses = bayes.loadDataSet()
# myVoacbList = bayes.createVocabList(listOPosts)
# returnVec = bayes.setOfWords2Vec(myVoacbList, listOPosts[0])
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVoacbList, postinDoc))
#
# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
# print('%s\n%s\n%s' % (p0V, p1V, pAb))
# print(returnVec)
# print(sorted(myVoacbList))
