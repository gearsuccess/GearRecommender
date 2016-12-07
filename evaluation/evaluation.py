#-*- coding: UTF-8 -*-
from __future__ import division
from numpy import *
import math
import pandas as pd
#from DataModel import FileDataModel


class Eval:
    """对算法进行评估。

    根据算法的推荐结果和真实值，计算算法的F1值和NDCG等。
    """
    def __init__(self):
       pass


    def RMSE(self, predict_rating_list, true_rating_list):
        error = array([(predict_rating_list[i]-true_rating_list[i])**2 for i in range(len(predict_rating_list))])
        reslut = math.sqrt(error.sum()/len(error))
        return reslut

    def F1_score_Hit_ratio(self, recommend_list, purchased_list):
        """计算F1值。

        输入：
            recommend_list： 推荐算法给出的推荐结果。
            purchased_list： 用户的真实购买记录。

        输出：
            F1值。
        """
        user_number = len(recommend_list)
        correct = []
        co_length = []
        re_length = []
        pu_length = []
        p = []
        r = []
        f = []
        hit_number = 0
        for i in range(user_number):
            temp = []
            for j in recommend_list[i]:
                if j in purchased_list[i]:
                    temp.append(j)
            if len(temp):
                hit_number = hit_number + 1
            co_length.append(len(temp))
            re_length.append(len(recommend_list[i]))
            pu_length.append(len(purchased_list[i]))
            correct.append(temp)

        for i in range(user_number):
            if re_length[i] == 0:
                p_t = 0
            else:
                p_t = co_length[i] / re_length[i]
            if pu_length[i] == 0:
                r_t = 0
            else:
                r_t = co_length[i] / pu_length[i]
            p.append(p_t)
            r.append(r_t)
            if p_t != 0 or r_t != 0:
                f.append(2*p_t*r_t / (p_t+r_t))
            else:
                f.append(0)

        hit_ratio = hit_number / user_number
        return f, p, r, hit_ratio

    def NDGG_k(self, recommend_list, purchased_list):
        """计算NDCG值。

        输入：
            recommend_list： 推荐算法给出的推荐结果。
            purchased_list： 用户的真实购买记录。

        输出：
            NDCG值。
        """
        user_number = len(recommend_list)
        u_ndgg = []
        for i in range(user_number):
            temp = 0
            Z_u = 0
            for j in range(len(recommend_list[i])):
                Z_u = Z_u + 1 / log2(j + 2)
                if recommend_list[i][j] in purchased_list[i]:
                    temp = temp + 1 / log2(j + 2)
            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            u_ndgg.append(temp)
        NDCG = array(u_ndgg).mean()
        return NDCG

    def evalAll(self,recommend_list, purchased_list):
        """计算F1和NDCG值。

        输入：
            recommend_list： 推荐算法给出的推荐结果。
            purchased_list： 用户的真实购买记录。

        输出：
            F1，NDCG值。
        """
        f1, p, r, hit_ratio = self.F1_score_Hit_ratio(recommend_list, purchased_list)
        NDCG = self.NDGG_k(recommend_list, purchased_list)
        return array(f1).mean(), hit_ratio, NDCG, array(p).mean(), array(r).mean()



