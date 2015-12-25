# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:01:08 2015

@author: gajendrakatuwal
"""

from scipy.stats import ttest_ind
from math import sqrt
import numpy as np
import pandas as pd

def cohens_d(x0, x1, pooled_sd=True):
    # print('pooled_sd = True ..............')
    dmean = np.mean(x0) - np.mean(x1)
    n0 = len(x0)
    n1 = len(x1)
    #print(n0, n1)
    if pooled_sd:
        sd = sqrt((  (np.std(x0) ** 2)*(n0-1) + (np.std(x1) ** 2)*(n1-1)) / (n0 + n1 -2))
        sd1 = sqrt((np.std(x0) ** 2 + np.std(x1) ** 2) / 2)
    #print(sd, sd1)
    return dmean/sd

def get_stats(x, label):
    """
    x1:nd array
    x2: nd array
    label: nd array
    
    returns mean, std, t
    """
    unique_labels = np.unique(label)  # orders as 0, 1
    x0 = x[label == unique_labels[0]]
    x1 = x[label == unique_labels[1]]
    
    mean_x0 = np.mean(x0)
    mean_x1 = np.mean(x1)
    std_x0 = np.std(x0)
    std_x1 = np.std(x1)
    t_val, p_val = ttest_ind(x0, x1,  axis=0, equal_var=False)
    d = cohens_d(x0, x1)
    stats = pd.Series({'mean_x0': mean_x0, 'mean_x1': mean_x1, 'std_x0': std_x0, 'std_x1': std_x1, 't_val': t_val, 'p_val': p_val, 'd': d})
    # stats = stats.apply(lambda x: round(x, 4))
    return(stats)

def remove_highly_correlated(df, method='pearson', threshold=0.95):
    """
    remove highly correlated features before classification
    """
    df_corr = df.corr(method=method)
    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col], drops):
           continue
        # find all the variables that are highly correlated with the current variable 
        # and add them to the drop list 
        corr = df_corr[abs(df_corr[col]) > threshold].index
        drops = np.union1d(drops, corr)
    logger.info("Dropping {} highly correlated features...".format(drops.shape[0]))
    return (df.drop(drops, axis=1))
