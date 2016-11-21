import pickle
import numpy as np
import logging
import logging.config
import logging.handlers
from evaluation.evaluation import *

class BPR():
    def __init__(self, path, parameters):
        self.load_data(path)
        self.factors = parameters['factors']
        self.learning_rate = parameters['learning_rate']
        self.bias_regularization = parameters['bias_regularization']
        self.user_regularization = parameters['user_regularization']
        self.positive_item_regularization = parameters['positive_item_regularization']
        self.negative_item_regularization = parameters['negative_item_regularization']
        self.iter = parameters['iter']
        self.path = path
        self.TopN = parameters['n']
        self.recommend_new = parameters['recommend_new']
        self.insights = parameters['display']
        self.number_of_test_seen = parameters['number_of_test_seen']

        logging.config.fileConfig('log_conf')
        self.bpr_logger = logging.getLogger('bpr')
        self.bpr_logger.info(''.join(('Factors: ', str(self.factors), ' learningrate: ', str(self.learning_rate),
                             ' bias_regularization: ', str(self.bias_regularization), ' user_regularization: ',
                             str(self.user_regularization), ' positive_item_regularization: ', str(self.positive_item_regularization),
                            ' negative_item_regularization: ', str(self.negative_item_regularization), ' iter:', str(self.iter),
                            ' TopN:', str(self.TopN))))


    def load_data(self, path):
        print path

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

    def update_factors(self, u, i, j, update_u=True, update_i=True):
        """apply SGD update"""
        update_j = True
        x = self.item_bias[i] + np.dot(self.user_factors[u], self.item_factors[i]) \
            - self.item_bias[j] - np.dot(self.user_factors[u], self.item_factors[j])

        z = 1.0/(1.0+exp(x))
        # update bias terms

        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i]-self.item_factors[j])*z - self.user_regularization*self.user_factors[u]
            self.user_factors[u] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u]*z - self.positive_item_regularization*self.item_factors[i]
            self.item_factors[i] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u]*z - self.negative_item_regularization*self.item_factors[j]
            self.item_factors[j] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0
        for u, i, j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            ranking_loss += 1.0 / (1.0+exp(x))

        complexity = 0
        for u, i, j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2
        return ranking_loss + 0.5*complexity

    def fit(self):

        temp = math.sqrt(self.factors)
        self.item_bias = np.zeros(self.item_count)
        self.user_factors = np.array([[(0.1 * random.random() / temp) for j in range(self.factors)] for i in range(self.user_count)])
        self.item_factors = np.array([[(0.1 * random.random() / temp) for j in range(self.factors)] for i in range(self.item_count)])

        self.loss_sample_number = 100000
        self.loss_samples = []
        for num in range(self.loss_sample_number):
            u = random.randint(0,self.user_count-1)
            if self.user_purchased_item_dict.has_key(u):
                i = random.choice(self.user_purchased_item_dict[u])
                j = random.randint(0, self.item_count-1)
                while j in self.user_purchased_item_dict[u]:
                    j = random.randint(0, self.item_count - 1)
                self.loss_samples.append([u, i, j])

        self.update_samples = []
        for u in range(self.user_count):
            if self.user_purchased_item_dict.has_key(u):
                i = random.choice(self.user_purchased_item_dict[u])
                j = random.randint(0, self.item_count - 1)
                while j in self.user_purchased_item_dict[u]:
                    j = random.randint(0, self.item_count - 1)
                self.update_samples.append([u, i, j])

        old_loss = float('Inf')

        for it in xrange(self.iter):
            self.bpr_logger.info('starting iteration {0}'.format(it))
            for u, i, j in self.update_samples:
                self.update_factors(u, i, j)

            if (it+1) % self.number_of_test_seen == 0:
                current_loss = self.score(1)[1]
            else:
                current_loss = self.score(0)

            if current_loss - old_loss > 0:
                self.bpr_logger.info('converge!!')
                break
            else:
                old_loss = current_loss
                self.learning_rate *= 0.9


    def save(self):
        t = pd.DataFrame(self.item_bias)
        t.to_csv('../results/bpr_item_bias')
        t = pd.DataFrame(self.user_factors)
        t.to_csv('../results/bpr_user_factors')
        t = pd.DataFrame(self.item_factors)
        t.to_csv('../results/bpr_item_factors')
        t = pd.DataFrame(self.loss_samples)
        t.to_csv('../results/bpr_loss_samples')
        t = pd.DataFrame(self.user_recommend)
        t.to_csv('../results/bpr_user_recommend')


    def predict(self,user,item):
        result = self.item_bias[item] + np.dot(self.user_factors[user], self.item_factors[item])
        if result > 5:
            return 5
        else:
            if result < 1:
                return 1
            else:
                return result

    def recommend(self, u):
        if self.recommend_new == 0:
            candidate = np.array([self.predict(u, i) for i in range(self.item_count)])
        else:
            candidate = np.array([self.predict(u, i) for i in range(self.item_count) if i not in self.user_purchased_item_dict[u]])

        result = np.argsort(candidate)[-1:-self.TopN-1:-1]
        return result

    def score(self, final_score):
        current_loss = self.loss()
        if final_score:
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
            self.bpr_logger.info(','.join(('test:', 'f1:'+str(f1), 'hit:'+str(hit), 'ndcg:'+str(ndcg), 'p:'+str(p), 'r:'+str(r) )))
            return [f1, current_loss, rmse, ndcg, p, r]
        else:
            self.bpr_logger.info('training loss: ' + str(current_loss))
            return current_loss




