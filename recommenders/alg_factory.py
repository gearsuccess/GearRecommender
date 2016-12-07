from recommenders.collabrativefiltering.biased_mf import BiasedMF
from recommenders.collabrativefiltering.bpr import BPR
from recommenders.collabrativefiltering.item_cf import ItemCF
from recommenders.collabrativefiltering.user_cf import UserCF
from recommenders.static.average import AVERAGE
from recommenders.static.top_n import TopN


class AlgFactory():

    @staticmethod
    def create(name, path, parameters):
        if name == 'TopN':
            return TopN(path, parameters)
        elif name == 'BiasedMF':
            return BiasedMF(path, parameters)
        elif name == 'AVERAGE':
            return AVERAGE(path, parameters)
        elif name == 'UserCF':
            return UserCF(path, parameters)
        elif name == 'ItemCF':
            return ItemCF(path, parameters)
        elif name == 'BPR':
            return BPR(path, parameters)

