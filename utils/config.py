#-*- coding: UTF-8 -*-
import pandas as pd
import ConfigParser
import numpy as np

class Parameter:
    '''
    算法参数类。
    对算法的参数进行封装，方便调参。
    '''
    def __init__(self, paras):
        self.paras = paras
        self.len_list = {}
        self.current_pos = {}
        self.para_dict_list = []
        for k, values in paras.items():
            self.len_list[k] = len(values)
            self.current_pos[k] = 0

    def get_paramter_list(self, method, number_of_sample=3):
        if method == 'grid':
            self._gen_paramter_list_grid(0)
        elif method == 'random':
            self._gen_paramter_list_random(number_of_sample)

        self.para_dict_list = [self.to_dict(i) for i in self.para_dict_list]
        return self.para_dict_list

    def _gen_paramter_list_grid(self, n):
        number_of_parameters = len(self.len_list.keys())
        if(n == number_of_parameters):
            return
        else:
            key_name = self.paras.keys()[n]
            for i in range(self.len_list[key_name]):
                self._gen_paramter_list_grid(n+1)
                if self.current_pos.values() not in self.para_dict_list:
                    self.para_dict_list.append(self.current_pos.values())
                self.current_pos[key_name] += 1
            self.current_pos[key_name] = 0

    def _gen_paramter_list_random(self, n):
        max = 1
        for k, values in self.paras.items():
            max = max * self.len_list[k]
        if max < n:
            self._gen_paramter_list_grid(0)
        else:
            for num in range(n):
                tmp = [np.random.randint(0, self.len_list[self.paras.keys()[i]]) for i in range(len(self.paras.keys()))]
                self.para_dict_list.append(tmp)

    def to_dict(self, pos):
        '''
        将参数表示成字典类型。
        '''
        result = {}
        for i in range(len(self.paras.keys())):
            k = self.paras.keys()[i]
            v = pos[i]
            result[k] = self.paras[k][v]
        return result

    def __iter__(self):
        return self


class Config:
    '''
    读取配置文件。
    '''
    def __init__(self, config_path):
        self.alg_parameters = {}
        self.general_settings = {}
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(config_path)


    def _get_alg_parameters(self, split_string='|'):
        self.alg_list = self.conf.get('algorithms', 'all').split(split_string)
        self.secs = self.conf.sections()
        for alg in self.alg_list:
            if alg in self.secs:
                para = {}
                opts = self.conf.options(alg)
                for opt in opts:
                    para[opt] = eval(self.conf.get(alg, opt))
                self.alg_parameters[alg] = Parameter(para)
        return self.alg_parameters

    def _get_general_settings(self, split_string='|'):
        gs = self.conf.options('general')
        for i in gs:
            self.general_settings[i] = self.conf.get('general', i)
        return self.general_settings

    def get_all(self):
        self._get_alg_parameters()
        self._get_general_settings()

if __name__ == '__main__':

    config = Config('../Main/conf')
    print config._get_general_settings()

