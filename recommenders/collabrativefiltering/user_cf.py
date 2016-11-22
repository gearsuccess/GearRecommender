#-*- coding: UTF-8 -*-
import pickle
import numpy as np
import logging
import logging.handlers
from utils.similarity import Similarity
from sklearn.base import BaseEstimator
from evaluation.evaluation import *

class UserCF(BaseEstimator):
    """基于用户的协同过滤算法。

    基于用户对物品的历史交互行为，计算用户之间的相似度，然后基于该相似度对用户进行推荐。

    属性：
        neighbornum： 近邻数量。
        similarity： 相似度度量方法。
        n： 推荐结果的数量。
        dataModel： 数据模型。
        simiMatrix：相似度矩阵。
    """

    def __init__(self, path, parameters):
        self.user_index_dict = pickle.load(open(path + 'uiDict', 'rb'))
        self.item_index_dict = pickle.load(open(path + 'iiDict', 'rb'))
        #self.train_data = pd.read_csv(path + 'eccTrainData')
        self.user_purchased_item_dict = pickle.load(open(path + 'upiTrainDict', 'rb'))
        self.item_purchased_user_dict = pickle.load(open(path + 'ipuTrainDict', 'rb'))
        self.user_item_rating_dict = pickle.load(open(path + 'uiraTrainDict', 'rb'))

        self.true_rating_dict = pickle.load(open(path + 'uiraTestDict', 'rb'))
        self.true_purchased_dict = pickle.load(open(path + 'upiTestDict', 'rb'))

        self.user_count = len(self.user_index_dict.keys())
        self.item_count = len(self.item_index_dict.keys())

        print self.user_count, self.item_count


        self.similarity = Similarity('COSINE')
        self.path = path
        self.top_n = parameters['n']
        self.neighbornum = parameters['neighbornum']
        self.similarity = Similarity('COSINE')
        self.recommend_new = parameters['recommend_new']

        logging.config.fileConfig('log_conf')
        self.usercf_logger = logging.getLogger('usercf')
        self.usercf_logger.info(''.join(('similarity: ', str(self.similarity), ' neighbornum: ', str(self.neighbornum),
                             ' TopN:', str(self.top_n))))


    def fit(self):
        usersNum = self.user_count
        self.simiMatrix = np.zeros((usersNum, usersNum))
        self.usercf_logger.info('computing similarity ...')
        for i in range(usersNum):
            if self.user_purchased_item_dict.has_key(i):
                for j in range(i + 1, usersNum):
                    if self.user_purchased_item_dict.has_key(j):
                        s = self.similarity.compute(self.user_purchased_item_dict[i], self.user_purchased_item_dict[j])
                        self.simiMatrix[i][j] = self.simiMatrix[j][i] = s
                    else:
                        self.simiMatrix[i][j] = self.simiMatrix[j][i] = 0.0
            else:
                for j in range(i + 1, usersNum):
                    self.simiMatrix[i][j] = self.simiMatrix[j][i] = 0.0

    def save(self):
        t = pd.DataFrame(self.user_recommend)
        t.to_csv('../results/user_cf_user_recommend')

    def predict(self, user, item):
        rating = 0.0
        sum = 0.0
        neighbors = np.argsort(np.array(self.simiMatrix[user]))[-1:-self.neighbornum-1:-1]
        for i in neighbors:
            if self.user_purchased_item_dict.has_key(i) and item in self.user_purchased_item_dict[i]:
                index = str(i) + '##' + str(item)
                rating += self.simiMatrix[user][i] * self.user_item_rating_dict[index]
                sum += self.simiMatrix[user][i]
        if sum != 0:
            rating = rating / sum
        if rating > 5:
            return 5
        else:
            if rating < 1:
                return 1
            else:
                return rating

    def recommend(self, u):
        candidate_ratings = np.array([self.predict(u, i) for i in range(self.item_count)])
        candidate_items = np.argsort(candidate_ratings)[-1::-1]
        if self.recommend_new == 0:
            result = candidate_items[:self.TopN]
        else:
            new_items = np.array([i for i in candidate_items if i not in self.user_purchased_item_dict[u]])
            result = new_items[:self.TopN]
        return result

    def score(self, log):
        e = Eval()
        predict_rating_list = []
        true_rating_list = []
        predict_top_n = []
        true_purchased = []
        self.user_recommend = []

        for (ui, rating) in self.true_rating_dict.items():
            user = int(ui.split('##')[0])
            item = int(ui.split('##')[1])
            predict_rating_list.append(self.predict(user, item))
            true_rating_list.append(rating)

        for (u, items) in self.true_purchased_dict.items():
            recommended_item = self.recommend(u)
            predict_top_n.append(recommended_item)
            self.user_recommend.append([u, recommended_item])
            true_purchased.append(items)

        rmse = e.RMSE(predict_rating_list, true_rating_list)
        f1, hit, ndcg, p, r = e.evalAll(predict_top_n, true_purchased)
        self.usercf_logger.info(','.join(('f1:'+str(f1), 'hit:'+str(hit), 'ndcg:'+str(ndcg), 'p:'+str(p), 'r'+str(r) )))
        return [rmse, f1, hit, ndcg, p, r]


