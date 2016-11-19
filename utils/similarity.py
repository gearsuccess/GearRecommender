#-*- coding: UTF-8 -*-
import math


class Similarity:
    """相似度度量。

    实现了各种相似度度量算法，包括常见的余弦相似度，jaccard距离等。

    """
    def __init__(self, simiName):
        self.simiName = simiName

    @staticmethod
    def cosine(v1, v2):
        """计算余弦相似度。

        输入：
            v1： 向量1.
            v2： 向量2.

        输出：
            两个向量的余弦相似度。
        """
        if len(v1)==0 or len(v2) == 0:
            return 0
        else:
            return len(set(v1) & set(v2)) / math.sqrt(len(v1) * len(v2) * 1.0)

    @staticmethod
    def jaccard(v1, v2):
        """计算jaccard相似度。

        输入：
            v1： 向量1.
            v2： 向量2.

        输出：
            两个向量的jaccard相似度。
        """
        s1 = set(v1)
        s2 = set(v2)
        return len(s1.intersection(s2)) * 1.0 / len(s1.union(s2))

    def compute(self, v1, v2):
        return SIMILARITY.get(self.simiName)(v1, v2)

SIMILARITY = {'COSINE': Similarity.cosine, 'JACCARD': Similarity.jaccard}

if __name__ == "__main__":
    print Similarity.cosine([1,2], [2, 3])
    print Similarity.jaccard([1,2], [2, 3])
    s = Similarity('COSINE')
    print s.compute([1, 2], [7, 2])