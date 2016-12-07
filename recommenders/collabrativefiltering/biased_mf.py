import datetime
from evaluation.evaluation import *
from visualization.training_process import *
import pickle
import logging
import logging.handlers


class BiasedMF():
    def __init__(self, path, parameters):
        random.seed(10)
        np.random.seed = 10
        self.load_data(path)
        self.factors = parameters['factors']
        self.learning_rate = parameters['learningrate']
        self.user_regular = parameters['userregular']
        self.item_regular = parameters['itemregular']
        self.iter = parameters['iter']
        self.TopN = parameters['n']
        self.recommend_new = parameters['recommend_new']
        self.visualization = parameters['visualization']
        self.number_of_test_seen = parameters['number_of_test_seen']
        self.main_evaluation = parameters['main_evaluation']

        self.results = []
        logging.config.fileConfig('log_conf')
        self.biasedMF_logger = logging.getLogger('biasedMF')
        self.biasedMF_logger.info('Factors: '+str(self.factors) +' learningrate: '+ str(self.learning_rate)+
                                  ' userregular: '+str(self.user_regular) + ' itemregular: '+str(self.item_regular)+
                                  ' iter:' + str(self.iter)+' TopN:' + str(self.TopN))

    def load_data(self, path):
        self.user_index_dict = pickle.load(open(path + 'uiDict', 'rb'))
        self.item_index_dict = pickle.load(open(path + 'iiDict', 'rb'))

        self.user_purchased_item_dict = pickle.load(open(path + 'upiTrainDict', 'rb'))
        self.item_purchased_user_dict = pickle.load(open(path + 'ipuTrainDict', 'rb'))
        self.user_item_rating_dict = pickle.load(open(path + 'uiraTrainDict', 'rb'))

        self.true_rating_dict = pickle.load(open(path + 'uiraTestDict', 'rb'))
        self.true_purchased_dict = pickle.load(open(path + 'upiTestDict', 'rb'))

        self.user_count = len(self.user_index_dict.keys())
        self.item_count = len(self.item_index_dict.keys())
        print self.user_count, self.item_count

    def fit(self):
        self.mu = np.array([r for (ui, r) in self.user_item_rating_dict.items()]).mean()
        self.mu = np.array([r for (ui, r) in self.user_item_rating_dict.items()]).mean()
        temp = math.sqrt(self.factors)
        self.bu = np.array([random.random() for i in range(self.user_count)])
        self.bi = np.array([random.random() for i in range(self.item_count)])
        self.pu = np.array([np.array([(0.1*random.random() / temp) for j in range(self.factors)]) for i in range(self.user_count)])
        self.qi = np.array([np.array([(0.1*random.random() / temp) for j in range(self.factors)]) for i in range(self.item_count)])
        self.pre_loss = float('Inf')


        for step in range(self.iter):
            self.biasedMF_logger.info('iteration: ' + str(step))
            s = datetime.datetime.now()
            for (ui, r) in self.user_item_rating_dict.items():
                user = int(ui.split('##')[0])
                item = int(ui.split('##')[1])
                eui = 5*r - self.predict(user, item)
                self.bu[user] += self.learning_rate*(eui-self.user_regular*self.bu[user])
                self.bi[item] += self.learning_rate*(eui-self.item_regular*self.bi[item])
                temp = self.qi[item]
                self.qi[item] += self.learning_rate*(np.dot(eui, self.pu[user]) - np.dot(self.item_regular, self.qi[item]))
                self.pu[user] += self.learning_rate*(np.dot(eui, temp) - np.dot(self.user_regular, self.pu[user]))
            e = datetime.datetime.now()
            self.biasedMF_logger.info('iteration time cost: ' + str(e-s))

            current_loss = self.loss()
            self.biasedMF_logger.info('training loss: ' + str(current_loss))

            if (step+1) % self.number_of_test_seen == 0:
                self.score()
                self.save(str(step+1))

            if current_loss > self.pre_loss or abs(current_loss - self.pre_loss) < 0.01:
                self.biasedMF_logger.info('converge!!')
                break
            else:
                self.pre_loss = current_loss
                self.learning_rate = self.learning_rate * 0.93

    def save(self, itereation):
        t = pd.DataFrame([self.mu])
        t.to_csv('../results/biased_mf_mu'+itereation)
        t = pd.DataFrame(self.pu)
        t.to_csv('../results/biased_mf_pu'+itereation)
        t = pd.DataFrame(self.qi)
        t.to_csv('../results/biased_mf_qi'+itereation)
        t = pd.DataFrame(self.bu)
        t.to_csv('../results/biased_mf_bu'+itereation)
        t = pd.DataFrame(self.bi)
        t.to_csv('../results/biased_mf_bi'+itereation)
        t = pd.DataFrame(self.results)
        t.to_csv('../results/results'+itereation)
        t = pd.DataFrame(self.trec_output)
        t.to_csv('../results/biased_mf_trec_output'+itereation, sep=' ', index=False, header=False)

    def predict(self, user, item):
        ans = self.mu + self.bi[item] + self.bu[user] + np.dot(self.qi[item], self.pu[user])
        '''
        if ans > 5:
            return 5
        elif ans < 1:
            return 1
        '''
        return ans

    def recommend(self, u):
        candidate_ratings = np.array([self.predict(u, i) for i in range(self.item_count)])
        candidate_items = np.argsort(candidate_ratings)[-1::-1]
        if self.recommend_new == 0:
            result = candidate_items[:self.TopN]
            result_ratings = candidate_ratings[result]
        else:
            new_items = np.array([i for i in candidate_items if i not in self.user_purchased_item_dict[u]])
            result = new_items[:self.TopN]
            result_ratings = candidate_ratings[result]
        return result, result_ratings

    def loss(self):
        current_loss = 0.0
        for (ui, r) in self.user_item_rating_dict.items():
            user = int(ui.split('##')[0])
            item = int(ui.split('##')[1])
            eui = 5*r - self.predict(user, item)
            current_loss += eui ** 2
        for user in self.user_index_dict.keys():
            user = int(user)
            current_loss += self.user_regular * (np.dot(self.pu[user], self.pu[user]) + self.bu[user] ** 2)
        for item in self.item_index_dict.keys():
            item = int(item)
            current_loss += self.item_regular * (np.dot(self.qi[item], self.qi[item]) + self.bi[item] ** 2)
        return current_loss

    def score(self):
        e = Eval()
        predict_rating_list = []
        true_rating_list = []
        predict_top_n = []
        true_purchased = []
        self.trec_output = []

        for (ui, rating) in self.true_rating_dict.items():
            user = int(ui.split('##')[0])
            item = int(ui.split('##')[1])
            predict_rating_list.append(self.predict(user, item))
            true_rating_list.append(rating)

        for (u, items) in self.true_purchased_dict.items():
            recommended_item, recommended_item_ratings = self.recommend(u)
            predict_top_n.append(recommended_item)
            for i in range(len(recommended_item)):
                if recommended_item[i] in items:
                    row = [self.user_index_dict[u],'Q0',self.item_index_dict[recommended_item[i]],i+1,recommended_item_ratings[i], 'BiasedMF', 'right']
                else:
                    row = [self.user_index_dict[u],'Q0',self.item_index_dict[recommended_item[i]],i+1,recommended_item_ratings[i], 'BiasedMF', 'wrong']
                self.trec_output.append(row)
            true_purchased.append(items)

        rmse = e.RMSE(predict_rating_list, true_rating_list)
        f1, p, r, hit_ratio = e.F1_score_Hit_ratio(predict_top_n, true_purchased)
        ndcg = e.NDGG_k(predict_top_n, true_purchased)

        self.results.append([rmse, f1, p, r, hit_ratio, ndcg])
        self.biasedMF_logger.info(','.join(('test:', 'f1:'+str(array(f1).mean()), 'hit:'+str(hit_ratio), 'ndcg:'+str(ndcg), 'p:'+str(array(p).mean()), 'r:'+str(array(r).mean()), 'rmse:' + str(rmse))))
        return [rmse, array(f1).mean(), array(p).mean(), array(r).mean(), hit_ratio, ndcg]










