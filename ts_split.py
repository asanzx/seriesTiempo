import numpy as np
import pandas as pd

def ts_split(X, y, cv=4, test_split=0.3, sliding=False):
    '''
    ts_split()[a][b][c]
    a: cv fold (1 <= a <= cv)
    b: 0-Train / 1-Test
    c: 0-X / 1-y
    '''
    train_test_df_dict = {}
    temp = 0
    
    for i in range(cv):
        group = (i+1)*int(X.shape[0]/cv)
        group_split = int(group*(1 - test_split))
        train_df = [X[:group_split,:], y[:group_split]]
        test_df = [X[group_split:group,:], y[group_split:group]]
        
        if i == cv-1:
            train_df = [X[:group_split,:], y[:group_split]]
            test_df = [X[group_split:X.shape[0],:], y[group_split:X.shape[0]]]
        
        train_test_df_dict[i+1] = [train_df, test_df]
    
    if sliding:
        for i in range(cv):
            group = (i+1)*int(X.shape[0]/cv)
            group_split = int(X.shape[0]*(1-test_split)/cv) + temp
            train_df = [X[temp:group_split,:], y[temp:group_split]]
            test_df = [X[group_split:group,:], y[group_split:group]]
            
            if i == cv-1:
                train_df = [X[temp:group_split,:], y[temp:group_split]]
                test_df = [X[group_split:X.shape[0],:], y[group_split:X.shape[0]]]
            
            temp = group
            train_test_df_dict[i+1] = [train_df, test_df]
    
    return train_test_df_dict
