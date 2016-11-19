#-*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import scipy.sparse as spr
import os
from BaseDataModel import BaseDataModel

class MemeryDataModel(BaseDataModel):
    """数据模型类。

    对数据进行建模，将按行记录的原始数据保存成稀疏矩阵格式，一方面方便算法进行高效访问，
    另一方面，由于用户物品交互记录总是稀疏的，用稀疏矩阵的格式可以节省空间。

    属性：
        samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
        targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。
        __users： 训练集中包含的用户列表。
        __items： 训练集中包含的物品列表。
        ratingMatrix：训练数据的矩阵表示。
        __data：ratingMatrix表示成稀疏矩阵的形式。
        __data_T：ratingMatrix的转置的稀疏矩阵形式。
    """
    def __init__(self, samples, targets, isRating=True, hasTimes=False):
        """构造函数。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。
            isRating： 指示训练数据是否包含评分，默认为True。
            hasTimes： 指示训练数据中是否包含行为次数，默认为False。
        """
        super(MemeryDataModel, self).__init__()
        self.samples = samples
        self.targets = targets
        self.__users = list(set(np.array(self.samples)[:,0]))
        self.__items = list(set(np.array(self.samples)[:,1]))
        self.ratingMatrix = np.zeros((len(self.__users),len(self.__items)))

        l = len(samples)
        for i in range(l):
            user = int(float(samples[i][0]))
            uid = self.getUidByUser(user)
            item = int(float(samples[i][1]))
            iid = self.getIidByItem(item)
            # 根据训练数据的类型，也就是传入的参数来决定评分值，如果是评分数据，就直接
            # 使用评分值，如果是无反馈数据，并且包含次数，那么使用次数作为评分值，如果
            # 是无反馈数据，且不包含次数，那么默认评分值为1。
            if isRating:
                rating = float(targets[i])
            elif hasTimes:
                rating += 1
            else:
                rating = 1
            self.ratingMatrix[uid][iid] = rating
        self.__data = spr.csr_matrix(self.ratingMatrix)
        self.__data_T = spr.csr_matrix(self.ratingMatrix.transpose())

    def getUidByUser(self, user):
        """根据用户名获取用户ID。

        输入：
            user： 用户名

        输出：
            用户ID。
        """
        if user not in self.__users:
            return -1
        else:
            uid = np.argwhere(np.array(self.__users) == user)[0][0]
        return uid

    def getIidByItem(self, item):
        """根据物品名获取物品ID。

        输入：
            items： 物品名

        输出：
            物品ID。
        """
        if item not in self.__items:
            return -1
        else:
            iid = np.argwhere(np.array(self.__items) == item)[0][0]
        return iid

    def getUserByUid(self, uid):
        """根据用户ID获取用户名。

        输入：
            uid： 用户ID

        输出：
            用户名。
        """
        return self.__users[uid]

    def getItemByIid(self, iid):
        """根据物品ID获取物品名。

        输入：
            iid： 物品ID

        输出：
            物品名。
        """
        return self.__items[iid]

    def getUsersNum(self):
        """获取训练集中用户的数量。

        输入：
            无

        输出：
            用户总数。
        """
        return len(self.__users)

    def getItemsNum(self):
        """获取训练集中物品的数量。

        输入：
            无

        输出：
            物品总数。
        """
        return len(self.__items)

    def getItemIDsFromUid(self, uid):
        """根据用户ID得到该用户有过交互的物品集。

        输入：
            uid： 用户ID。

        输出：
            物品ID集合。
        """
        return self.__data[uid].indices

    def getUserIDsFromIid(self, iid):
        """根据物品ID得到与该物品有过交互的用户集。

        输入：
            iid： 物品ID。

        输出：
            用户ID集合。
        """
        return self.__data_T[iid].indices

    def getItemIDsForEachUser(self):
        """获得每个用户有过交互行为的物品集。

        输入：
            无

        输出：
            每个用户对应的物品集构成的列表。
        """
        itemIDs = []
        for uid in range(len(self.__users)):
            itemIDs.append(self.__data[uid].indices)
        return itemIDs

    def getRating(self, userID, itemID):
        """得到该用户对该物品的评分值。

        输入：
            userID： 用户ID。
            itemID： 物品ID。

        输出：
            评分值。
        """
        return self.__data[userID, itemID]

    def getData(self):
        """获得用户物品评分稀疏矩阵。

        输入：
            无

        输出：
            用户物品评分稀疏矩阵。
        """
        return self.__data

    def getLineData(self):
        """按行保存完整的用户物品交互记录。

        输入：
            无

        输出：
            用户物品交互记录列表。
        """
        lineData = [[self.samples[i][0], self.samples[i][1], self.targets[i]] for i in range(len(self.samples))]
        return lineData


class MemeryDataModelPreprocess():
    """数据模型预处理类。

    对训练数据集进行预处理。

    属性：
        无
    """
    def __init__(self):
        print 'Begin Preprocess!'

    def getLineDataByRemoveDuplicate(self, samples, targets):
        """移除用户物品交互记录中的重复记录。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。

        输出：
            过滤后的用户物品交互记录列表。
        """
        result = []
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            print line
            if len(result) == 0:
                result.append(line)
            else:
                userItemPairs = [list(i) for i in np.array(result)[:,:2]]
                if line[:2] not in userItemPairs:
                    result.append(line)
        new_samples = [list(i) for i in np.array(result)[:,:2]]
        new_targets = list(np.array(result)[:,2])
        return new_samples, new_targets

    def getItemPurchasedNumDistribute(self, samples, targets):
        """得到训练集中的物品分布。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。

        输出：
            以字典的形式记录的物品分布，每一项表示一个物品对应的交互次数。
        """
        result = dict()
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            item = line[1]
            if result.has_key(item):
                result[item] += 1
            else:
                result[item] = 1
        return result

    def getUserPurchaseNumDistribute(self, samples, targets):
        """得到训练集中的用户分布。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。

        输出：
            以字典的形式记录的用户分布，每一项表示一个用户对应的交互次数。
        """
        result = dict()
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            user = line[0]
            if result.has_key(user):
                result[user] += 1
            else:
                result[user] = 1
        return result

    def hasLowFrequencyUser(self, samples, targets, n=5):
        """判断训练集中是否包含低频用户。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。
            n： 低频阈值。

        输出：
            如果训练集中包含至少一个低频用户，则返回True，否则，返回False。
        """
        UserF = self.getUserPurchaseNumDistribute(samples, targets)
        f = UserF.values()
        for i in f:
            if i < n:
                return 1
        return 0

    def hasLowFrequencyItem(self, samples, targets, n=5):
        """判断训练集中是否包含低频物品。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。
            n： 低频阈值。

        输出：
            如果训练集中包含至少一个低频物品，则返回True，否则，返回False。
        """
        ItemF = self.getItemPurchasedNumDistribute(samples, targets)
        f = ItemF.values()
        for i in f:
            if i < n:
                return 1
        return 0

    def removeLowFrequencyUser(self, samples, targets, n=5):
        """移除训练集中的低频用户。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。
            n： 低频阈值。

        输出：
            过滤后的训练集和标签集。
        """
        print 'low'
        UserF = self.getUserPurchaseNumDistribute(samples, targets)
        result = []
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            user = line[0]
            if UserF[user] > n:
                result.append(line)
        new_samples = [list(i) for i in np.array(result)[:,:2]]
        new_targets = list(np.array(result)[:,2])
        return new_samples, new_targets

    def removeLowFrequencyItem(self, samples, targets, n=5):
        """移除训练集中的低频物品。

        输入：
            samples： 训练集，按行记录，每行表示一个用户对物品的交互行为。
            targets： 标签集，与训练集一一对应，表示每个训练样例对应的输出结果。
            n： 低频阈值。

        输出：
            过滤后的训练集和标签集。
        """
        print 'low'
        ItemF = self.getItemPurchasedNumDistribute(samples, targets)
        result = []
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            item = line[1]
            if ItemF[item] > n:
                result.append(line)
        new_samples = [list(i) for i in np.array(result)[:,:2]]
        new_targets = list(np.array(result)[:,2])
        return new_samples, new_targets



if __name__  ==  "__main__":
    data = pd.read_csv('../Data/bbg/transaction.csv')
    samples = [[int(i[0]), int(i[1])] for i in data.values[:, 0:2]]
    targets = [1 for i in samples]
    p = MemeryDataModelPreprocess()
    print p.hasLowFrequencyUser(samples, targets, 5)
