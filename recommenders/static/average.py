import numpy as np
import logging
import logging.handlers
from evaluation.evaluation import *
import time
import pickle

class AVERAGE():
    def __init__(self, path, parameters):
        self.user_index_dict = pickle.load(open(path + 'uiDict', 'rb'))
        self.item_index_dict = pickle.load(open(path + 'iiDict', 'rb'))
        self.train_data = pd.read_csv(path + 'eccTrainData')
        self.user_purchased_item_dict = pickle.load(open(path + 'upiTrainDict', 'rb'))
        self.item_purchased_user_dict = pickle.load(open(path + 'ipuTrainDict', 'rb'))
        self.user_item_rating_dict = pickle.load(open(path + 'uiraTrainDict', 'rb'))

        self.true_rating_dict = pickle.load(open(path + 'uiraTestDict', 'rb'))
        self.true_purchased_dict = pickle.load(open(path + 'upiTestDict', 'rb'))

        self.user_count = len(self.user_index_dict.keys())
        self.item_count = len(self.item_index_dict.keys())

        self.TopN = parameters['n']
        self.recommend_new = parameters['recommend_new']
        self.insights = parameters['display']

        logging.config.fileConfig('log_conf')
        self.average_logger = logging.getLogger('average')
        self.average_logger.info(''.join((' TopN:', str(self.TopN))))

    def fit(self):
        self.mu = np.array([r for (ui, r) in self.user_item_rating_dict.items()]).mean()

    def save(self):
        t = pd.DataFrame(self.user_recommend)
        t.to_csv('../results/average_user_recommend')

    def predict(self, user, item):
        ans = self.mu
        if ans > 5:
            return 5
        elif ans < 1:
            return 1
        return ans

    def recommend(self, u):
        if self.recommend_new == 0:
            candidate = np.array([self.predict(u, i) for i in range(self.item_count)])
        else:
            candidate = np.array([self.predict(u, i) for i in range(self.item_count) if i not in self.user_purchased_item_dict[u]])

        result = np.argsort(candidate)[-1:-self.TopN-1:-1]
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
        self.average_logger.info(','.join(('f1:'+str(f1), 'hit:'+str(hit), 'ndcg:'+str(ndcg), 'p:'+str(p), 'r'+str(r) )))
        return [rmse, f1, hit, ndcg, p, r]



