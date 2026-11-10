from math import log
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
import SUMethods.entropy_estimators as ee


def information_gain(f1, f2):
    """
    This function calculates the information gain, where ig(f1, f2) = H(f1) - H(f1\f2)

    :param f1: {numpy array}, shape (n_samples,)
    :param f2: {numpy array}, shape (n_samples,)
    :return: ig: {float}
    """

    ig = ee.entropyd(f1) - conditional_entropy(f1, f2)
    return ig


def conditional_entropy(f1, f2):
    """
    This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
    :param f1: {numpy array}, shape (n_samples,)
    :param f2: {numpy array}, shape (n_samples,)
    :return: ce {float} conditional entropy of f1 and f2
    """

    ce = ee.entropyd(f1) - ee.midd(f1, f2)
    return ce


def su_calculation(f1, f2):
    """
    This function calculates the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    :param f1: {numpy array}, shape (n_samples,)
    :param f2: {numpy array}, shape (n_samples,)
    :return: su {float} su is the symmetrical uncertainty of f1 and f2
    """
    # calculate information gain of f1 and f2, t1 = ig(f1, f2)
    # t1 = information_gain(f1, f2)
    # # calculate entropy of f1
    # t2 = ee.entropyd(f1)
    # # calculate entropy of f2
    # t3 = ee.entropyd(f2)
    # #if (t2 + t3) != 0:
    #
    # su = 2.0 * t1 / (t2 + t3)

    mutual_infor = mutual_info_classif(f1.reshape(-1, 1), f2, discrete_features=True)[0]
    feature_entropies = entropy(np.bincount(f1))
    label_entropy = entropy(np.bincount(f2))
    su = 2 * mutual_infor / (feature_entropies + label_entropy)
    return su

def mi_calculation(f1, f2):
    return ggsmi(f1, f2) #[f1, f2]


"""
计算Pheromone information
"""
def ggsmi(x, y):
    return ca_pab(x, y)*log((ca_pab(x, y)/(ee.entropyd(x)*ee.entropyd(y))))

"""
计算联合概率
"""
def ca_pab(x, y):
    return conditional_entropy(y, x)*ee.entropyd(x)

