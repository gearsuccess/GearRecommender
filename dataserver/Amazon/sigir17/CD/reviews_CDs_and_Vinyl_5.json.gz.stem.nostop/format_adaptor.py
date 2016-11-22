import pandas as pd
import numpy as np
import pickle

class Format():
    def __init__(self):
        print "begin"

    def generate_data(self):
        self.user_index_dict = dict()
        self.item_index_dict = dict()
        self.user_purchased_item_dict = dict()
        self.item_purchased_user_dict = dict()
        self.user_item_rating_dict = dict()

        self.true_rating_dict = dict()
        self.true_purchased_dict = dict()

        user_names = pd.read_csv("users.txt", names = ['name'])
        index = 0
        for line in user_names.values:
            self.user_index_dict[index] = line[0]
            index += 1


        item_names = pd.read_csv("product.txt", names = ['name'])
        index = 0
        for line in item_names.values:
            self.item_index_dict[index] = line[0]
            index += 1

        train = pd.read_csv("train_id.txt", sep='\t', names = ['uid', 'iid', 'reviewid'])
        for line in train.values:
            user_index = line[0]
            item_index = line[1]

            ui_index = str(user_index) + '##' + str(item_index)
            self.user_item_rating_dict[ui_index] = 1

            if user_index not in self.user_purchased_item_dict:
                self.user_purchased_item_dict[user_index] = [item_index]
            else:
                self.user_purchased_item_dict[user_index].append(item_index)

            if item_index not in self.item_purchased_user_dict:
                self.item_purchased_user_dict[item_index] = [user_index]
            else:
                self.item_purchased_user_dict[item_index].append(user_index)

        test = pd.read_csv("test_id.txt", sep='\t', names = ['uid', 'iid', 'reviewid'])
        for line in test.values:
            user_index = line[0]
            item_index = line[1]

            ui_index = str(user_index) + '##' + str(item_index)
            self.true_rating_dict[ui_index] = 1

            if user_index not in self.true_purchased_dict:
                self.true_purchased_dict[user_index] = [item_index]
            else:
                self.true_purchased_dict[user_index].append(item_index)

        pickle.dump(self.user_index_dict, open('..\uiDict', 'wb'))
        pickle.dump(self.item_index_dict, open('..\iiDict', 'wb'))
        pickle.dump(self.user_purchased_item_dict, open('..\upiTrainDict', 'wb'))
        pickle.dump(self.item_purchased_user_dict, open('..\ipuTrainDict', 'wb'))
        pickle.dump(self.user_item_rating_dict, open('..\uiraTrainDict', 'wb'))


        pickle.dump(self.true_rating_dict, open('..\uiraTestDict', 'wb'))
        pickle.dump(self.true_purchased_dict, open('..\upiTestDict', 'wb'))

    def test(self):
        self.user_index_dict = pickle.load(open('..\uiDict', 'rb'))
        self.item_index_dict = pickle.load(open('..\iiDict', 'rb'))
        self.user_purchased_item_dict = pickle.load(open('..\upiTrainDict', 'rb'))
        self.item_purchased_user_dict = pickle.load(open('..\ipuTrainDict', 'rb'))
        self.user_item_rating_dict = pickle.load(open('..\uiraTrainDict', 'rb'))


        self.true_rating_dict = pickle.load(open('..\uiraTestDict', 'r'))
        self.true_purchased_dict = pickle.load(open('..\upiTestDict', 'r'))

        print len(self.item_purchased_user_dict.keys())

if __name__ == "__main__":
    f = Format()
    f.generate_data()
    f.test()
