# -*- coding: UTF-8 -*-
import os
import datetime
import multiprocessing
import numpy as np
from recommenders.alg_factory import AlgFactory
from utils.config import Config
import random
import logging
import logging.config
import logging.handlers

class App:

    def __init__(self):
        self.alg_config = Config('conf')
        self.alg_config.get_all()
        self.path = self.alg_config.general_settings['inputpath']
        self.lock = multiprocessing.Lock()

    def __getstate__(self):
        d = dict(self.__dict__)
        return d

    def __setstate__(self, d):
        logging.config.fileConfig('log_conf')
        self.main_logger = logging.getLogger('main')
        self.__dict__.update(d)


    def mul_process(self, process_parameters):
        s = datetime.datetime.now()
        alg_name = process_parameters[0]
        parameters = process_parameters[1]

        if int(self.alg_config.general_settings['n_fold']) == 0:
            path = os.path.join(self.path, 'no_validation/')
            alg = AlgFactory.create(alg_name, path, parameters)
            alg.fit()
            main_evaluation = self.alg_config.general_settings['main_evaluation']
            r = np.array(alg.score(1)[main_evaluation])
            alg.save()
            e = datetime.datetime.now()
            self.lock.acquire()
            self.main_logger.info(str([alg_name+'on_test_set', r, parameters])+' training end! time cost:'+str(e-s))
            self.lock.release()
        else:
            r = np.zeros(int(self.alg_config.general_settings['n_fold']))
            for i in range(int(self.alg_config.general_settings['n_fold'])):
                path = os.path.join(self.path, str(i))
                alg = AlgFactory.create(alg_name, path, parameters)
                alg.fit()
                main_evaluation = self.alg_config.general_settings['main_evaluation']

                r = np.array(alg.score(1)[main_evaluation])
                alg.save()
            r = r.sum() / int(self.alg_config.general_settings['n_fold'])
            e = datetime.datetime.now()
            self.lock.acquire()
            self.main_logger.info(str([alg_name+'on_validation_set', r, parameters])+' training end! time cost:'+str(e-s))
            self.lock.release()

    def gear_go(self):
        process_que = []
        for alg_name in self.alg_config.alg_list:
            for para in self.alg_config.alg_parameters[alg_name].get_paramter_list('grid'):
                parameters = [alg_name, para]
                process = multiprocessing.Process(target=self.mul_process, args=(parameters,))
                process_que.append(process)
        for t in process_que:
            t.start()
        for t in process_que:
            t.join()

    def best_alg(self):
        return 0

    def recommend(self):
        return 0


if __name__ == '__main__':
    random.seed(1)
    np.random.seed = 1
    app = App()
    app.gear_go()