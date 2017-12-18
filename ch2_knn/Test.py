from numpy import *
import matplotlib.pyplot as plt
from ch2_knn import kNN



datingDataMat, datingLables = kNN.file2matrix('datedata\datingTestSet2.txt')


fig = plt.figure(figsize=(6, 6))
plt.subplot(111)
l1 = plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLables),15.0*array(datingLables) )

plt.ylim(-1,25)
plt.xlim(-5000,100000)
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')

#要显示三个图例就要三个scatter
plt.legend(l1, labels=['unlike', 'like', 'very like'], loc='upper left')

plt.show()

# for i in range(len(labels)):
#     if labels[i] == 1:  # 不喜欢
#         type1_x.append(matrix[i][0])
#         type1_y.append(matrix[i][1])
#
#     if labels[i] == 2:  # 魅力一般
#         type2_x.append(matrix[i][0])
#         type2_y.append(matrix[i][1])
#
#     if labels[i] == 3:  # 极具魅力
#         print i, '：', labels[i], ':', type(labels[i])
#         type3_x.append(matrix[i][0])
#         type3_y.append(matrix[i][1])
#
# type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
# type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
# type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
