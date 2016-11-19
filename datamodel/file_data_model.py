import numpy as np
import pandas as pd
import scipy.sparse as spr
import os
from BaseDataModel import BaseDataModel

class FileDataModelInRow(BaseDataModel):

    def __init__(self, config, isRating=True):
        super(FileDataModelInRow, self).__init__(config)
        train = pd.read_csv(self.conf.get('train'))
        test = pd.read_csv(self.conf.get('test'))

        self.__sum = 0.0

        self.__train = []
        self.__userIDsInTrain = set()
        for row in train.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            self.__sum += rating
            self.__train.append([uid, iid, rating])
            self.__userIDsInTrain.add(uid)

        self.__test = []
        self.__userIDsInTest = set()
        for row in test.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            self.__test.append([uid, iid, rating])
            self.__userIDsInTest.add(uid)

    def getUserIDsInTrain(self):
        return self.__userIDsInTrain

    def getUserIDsInTest(self):
        return self.__userIDsInTest

    def getUsersNumInTrain(self):
        return len(self.__userIDsInTrain)

    def getUsersNumInTest(self):
        return len(self.__userIDsInTest)

    def getItemIDsForEachUserInTest(self):
        itemIDs = [[] for i in self.__userIDsInTest]
        mapIDtoIndex = dict(zip(self.__userIDsInTest, range(self.getUsersNumInTest())))
        for row in self.__test:
            uid = row[0]
            iid = row[1]
            itemIDs[mapIDtoIndex.get(uid)].append(iid)
        return itemIDs

    def getTrain(self):
        return self.__train

    def getTest(self):
        return self.__test

    def getLenOfTrain(self):
        return len(self.__train)

    def getRow(self, n):
        return self.__train[n]

    def getAve(self):
        return self.__sum / self.getLenOfTrain()

class FileDataModelInMatrix(BaseDataModel):

    def __init__(self, config, isRating=True):
        super(FileDataModelInMatrix, self).__init__(config)
        self.__users = pd.read_csv(os.path.join(self.conf.get('dataset'), self.conf.get('users')))
        self.__items = pd.read_csv(os.path.join(self.conf.get('dataset'), self.conf.get('items')))
        train = pd.read_csv(os.path.join(self.conf.get('dataset'), self.conf.get('train')))
        test = pd.read_csv(os.path.join(self.conf.get('dataset'), self.conf.get('test')))
        ratingMatrixOfTrain = np.zeros((self.getUsersNum(), self.getItemsNum()))
        self.__userIDsInTrain = set()
        for row in train.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            ratingMatrixOfTrain[uid][iid] = rating
            self.__userIDsInTrain.add(uid)
        self.__train = spr.csr_matrix(ratingMatrixOfTrain)

        ratingMatrixOfTest = np.zeros((self.getUsersNum(), self.getItemsNum()))
        self.__userIDsInTest = set()
        for row in test.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            ratingMatrixOfTest[uid][iid] = rating
            self.__userIDsInTest.add(uid)
        self.__test = spr.csr_matrix(ratingMatrixOfTest)

        if self.getUsersNumInTrain() != self.getUsersNum():
            raise Exception('some users do not appear in train')

    def getUserIDsInTrain(self):
        return self.__userIDsInTrain

    def getUserIDsInTest(self):
        return self.__userIDsInTest

    def getUsersNumInTrain(self):
        return len(self.__userIDsInTrain)

    def getUsersNumInTest(self):
        return len(self.__userIDsInTest)

    def getItemIDsFromUserInTest(self, userID):
        return self.__Test[userID].indices

    def getItemIDsFromUserInTrain(self, userID):
        return self.__train[userID].indices

    def getItemIDsForEachUserInTrain(self):
        itemIDs = []
        for uid in self.__userIDsInTrain:
            itemIDs.append(self.__train[uid].indices)
        return itemIDs

    def getItemIDsForEachUserInTest(self):
        itemIDs = []
        for uid in self.__userIDsInTest:
            itemIDs.append(self.__test[uid].indices)
        return itemIDs

    def getRatingInTrain(self, userID, itemID):
        return self.__train[userID, itemID]

    def getTrain(self):
        return self.__train

    def getTest(self):
        return self.__test

    def getUserIDs(self):
        return self.__users.ix[:, 0]

    def getItemIDs(self):
        return self.__items.ix[:, 0]

    def getUserFromID(self, userID):
        return self.__users.ix[userID, 1]

    def getItemFromID(self, itemID):
        return self.__items.ix[itemID, 1]

    def getItemsNum(self):
        return self.__items.shape[0]

    def getUsersNum(self):
        return self.__users.shape[0]

    def getPopFile(self):
        return os.path.join(self.conf.get('dataset'), self.conf.get('pop'))


if __name__ == "__main__":
    pass