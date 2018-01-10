from ch4_bayes import bayes
import feedparser as fp
import json


ny = fp.parse('http://newyork.craigslist.org/stp/index.rss')
sf = fp.parse('http://sfbay.craigslist.org/stp/index.rss')
# ny = eval(open('ny2.txt', 'r').read())
# sf = eval(open('sf2.txt', 'r').read())
# ny = json.loads(open('ny2.txt', 'r').read())
# sf = json.loads(open('sf2.txt', 'r').read())
print(bayes.localWords(ny, sf))

# testDoc1 = ['love', 'my', 'dalmation']
# testDoc2 = ['dog', 'my', 'love']
# classLable = bayes.testingNB(testDoc2)
# print(classLable)
#
# errorrate = 0
# for i in range(100):
#     error = bayes.spamTest()
#     errorrate += error
#     #print(error)
# print(errorrate/100)
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
