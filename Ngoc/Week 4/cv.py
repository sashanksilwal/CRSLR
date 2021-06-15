"""
cv.py
Defines methods for conducting cross validation on a Bayesian Model.
Author: Ngoc Hoang
Last modified: June 15, 2021

Dependencies: the following packages and/or libraries are assumed
to have been imported prior to using the methods
- pandas
- pgmpy.models.BayesianModel
- pgmpy.estimators.BayesianEstimator
- sklearn.metrics.accuracy_score
"""

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from sklearn.metrics import accuracy_score

def partition(data, n):
    """Randomly partitions the data into n non-overlapping parts

    Args:
        data: pandasDataFrame object, the dataset to partition
        n: number of equal parts to split the data

    Returns:
        splits: list of n parts of the data, each one is a pandasDataFrame
    """
    splits = []
    remaining = data.copy(deep=True)
    for i in range(n):
        split = remaining.sample(frac=1/(n-i), random_state=10)
        splits.append(split)
        remaining = remaining.drop(split.index)
    return splits

def cross_val(model, data, n, target):
    """n-fold cross validation for a model

    Args:
        model: the estimator to train and validate
        data: pandasDataFrame object, the complete dataset
              it is assumed that the dataset contains the labels
        n: number of folds to conduct cross validation
        target: a string corresponding to the column that serves as labels
    
    Returns:
        scores: list of n elements corresponding to accuracy scores in each fold
    """
    scores = []
    splits = partition(data, n)
    for i in range(n):
        train_list = splits[:i] + splits[i+1:]
        train = pd.concat(train_list)
        test = splits[i]
        y_true = test[target]
        test = test.drop(columns=[target], axis=1)
        model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")
        y_pred = model.predict(test)
        acc = accuracy_score(y_pred[target], y_true)
        scores.append(acc)
    return scores