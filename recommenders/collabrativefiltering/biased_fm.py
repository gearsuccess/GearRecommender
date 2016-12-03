import datetime
from evaluation.evaluation import *
from display.training_process import *
import pickle
import logging
import logging.handlers
import copy

class BiasedFM():
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


        self.factors = parameters['factors']
        self.learning_rate = parameters['learningrate']
        self.user_regular = parameters['userregular']
        self.item_regular = parameters['itemregular']
        self.iter = parameters['iter']
        self.TopN = parameters['n']
        self.recommend_new = parameters['recommend_new']
        self.insights = parameters['display']
        self.number_of_test_seen = parameters['number_of_test_seen']
        self.main_evaluation = parameters['main_evaluation']

        self.figure_data = []
        logging.config.fileConfig('log_conf')
        self.biasedFM_logger = logging.getLogger('biasedFM')
        self.biasedFM_logger.info('Factors: '+str(self.factors) +' learningrate: '+ str(self.learning_rate)+
                                  ' userregular: '+str(self.user_regular) + ' itemregular: '+str(self.item_regular)+
                                  ' iter:' + str(self.iter)+' TopN:' + str(self.TopN))

    def fit(self):
        self.mu = np.array([r for (ui, r) in self.user_item_rating_dict.items()]).mean()
        self.mu = np.array([r for (ui, r) in self.user_item_rating_dict.items()]).mean()
        temp = math.sqrt(self.factors)
        self.bu = np.array([random.random() for i in range(self.user_count)])
        self.bi = np.array([random.random() for i in range(self.item_count)])
        self.pu = np.array([np.array([(0.1*random.random() / temp) for j in range(self.factors)]) for i in range(self.user_count)])
        self.qi = np.array([np.array([(0.1*random.random() / temp) for j in range(self.factors)]) for i in range(self.item_count)])

        self.pre_loss = float('Inf')
        self.figure_data.append([np.zeros(self.user_count),np.zeros(self.user_count),np.zeros(self.user_count),
                                 0, copy.deepcopy(self.pu),copy.deepcopy(self.qi),copy.deepcopy(self.bu),copy.deepcopy(self.bi)])


        for step in range(self.iter):
            self.biasedFM_logger.info('iteration: ' + str(step))
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
            self.biasedFM_logger.info('iteration time cost: ' + str(e-s))

            current_loss = self.loss()
            self.biasedFM_logger.info('training loss: ' + str(current_loss))

            if (step+1) % self.number_of_test_seen == 0:
                current_loss = self.score()


            if current_loss > self.pre_loss or abs(current_loss - self.pre_loss) < 0.01:
                self.biasedFM_logger.info('converge!!')
                break
            else:
                self.pre_loss = current_loss
                self.learning_rate = self.learning_rate * 0.93
        if step == self.iter - 1:
            self.biasedFM_logger.info('training end!')

        if self.insights:
            d = TrainingProcess(self.figure_data)
            d.run()

    def save(self):
        t = pd.DataFrame([self.mu])
        t.to_csv('../results/biased_fm_mu')
        t = pd.DataFrame(self.pu)
        t.to_csv('../results/biased_fm_pu')
        t = pd.DataFrame(self.qi)
        t.to_csv('../results/biased_fm_qi')
        t = pd.DataFrame(self.bu)
        t.to_csv('../results/biased_fm_bu')
        t = pd.DataFrame(self.bi)
        t.to_csv('../results/biased_fm_bi')
        t = pd.DataFrame(self.user_recommend)
        t.to_csv('../results/biased_fm_user_recommend')


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
        self.user_recommend = []
        trec_output = []

        for (ui, rating) in self.true_rating_dict.items():
            user = int(ui.split('##')[0])
            item = int(ui.split('##')[1])
            predict_rating_list.append(self.predict(user, item))
            true_rating_list.append(rating)

        for (u, items) in self.true_purchased_dict.items():
            recommended_item, recommended_item_ratings = self.recommend(u)
            predict_top_n.append(recommended_item)
            self.user_recommend.append([u, recommended_item])
            for i in range(len(recommended_item)):
                row = [self.user_index_dict[u],'Q0',self.item_index_dict[recommended_item[i]],i+1,recommended_item_ratings[i],'BPR']
                trec_output.append(row)
            true_purchased.append(items)
        t = pd.DataFrame(trec_output)
        t.to_csv('biasedMF_trec_output', sep=' ', index=False, header=False)

        rmse = e.RMSE(predict_rating_list, true_rating_list)
        f1, hit, ndcg, p, r = e.evalAll(predict_top_n, true_purchased)
        self.figure_data.append([np.array(p), np.array(r), rmse, copy.deepcopy(self.pu), copy.deepcopy(self.qi), copy.deepcopy(self.bu),copy.deepcopy(self.bi)])
        self.biasedFM_logger.info(','.join(('test:', 'f1:'+str(f1), 'hit:'+str(hit), 'ndcg:'+str(ndcg), 'p:'+str(p), 'r:'+str(r) )))
        return eval(self.main_evaluation)










