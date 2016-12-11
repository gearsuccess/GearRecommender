#-*- coding: UTF-8 -*-
import pickle
import numpy as np
import logging
import logging.handlers
from evaluation.evaluation import *


class TopN():
    def __init__(self, path, parameters):
        self.user_index_dict = pickle.load(open(path + 'uiDict', 'rb'))
        self.item_index_dict = pickle.load(open(path + 'iiDict', 'rb'))
        #self.train_data = pd.read_csv(path + 'eccTrainData')
        self.user_purchased_item_dict = pickle.load(open(path + 'upiTrainDict', 'rb'))
        self.item_purchased_user_dict = pickle.load(open(path + 'ipuTrainDict', 'rb'))

        self.true_rating_dict = pickle.load(open(path + 'uiraTestDict', 'rb'))
        self.true_purchased_dict = pickle.load(open(path + 'upiTestDict', 'rb'))

        self.user_count = len(self.user_index_dict.keys())
        self.item_count = len(self.item_index_dict.keys())

        print self.user_count, self.item_count


        self.n = parameters['n']
        self.recommend_new = parameters['recommend_new']
        self.main_evaluation = parameters['main_evaluation']

        logging.config.fileConfig('log_conf')
        self.topn_logger = logging.getLogger('topn')
        self.topn_logger.info(''.join((' TopN:', str(self.n))))

    def fit(self):
        item_itemnumber_list = [[i[0], len(i[1])] for i in self.item_purchased_user_dict.iteritems()]
        keys = sorted(item_itemnumber_list, key=lambda k: k[1], reverse = True)
        print keys
        self.popItems = np.array(keys)[:, 0]
        print self.popItems


    def save(self):
        t = pd.DataFrame(self.user_recommend)
        t.to_csv('../results/top_n_user_recommend')


    def predict(self, u, item):
        if item in self.popItems[:self.n]:
            return 5
        else:
            return 1

    def recommend(self, u):
        if self.recommend_new == 0:
            result = np.array(self.popItems[:self.n])
        else:
            result = np.array([i for i in self.popItems if i not in self.user_purchased_item_dict[u]])[:self.n]
        return result

    def score(self):
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
        self.topn_logger.info(','.join(('f1:'+str(f1), 'hit:'+str(hit), 'ndcg:'+str(ndcg), 'p:'+str(p), 'r'+str(r) )))
        return eval(self.main_evaluation)








