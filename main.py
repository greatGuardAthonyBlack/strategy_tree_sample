__author__ = 'Ricch'

import seaborn as sns
from matplotlib import pyplot as plt
from strategyTree import StrategyTree
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def discrete_data_set(df,discrete_df,threshold=10,method='qcut'):
    for c in df.columns[1:]:
        if len(df[c])<threshold:
            continue
        elif method=='cut':
            discrete_df[c]=pd.cut(df,threshold,labels=[i for i in range(threshold)])
        else:
            discrete_df[c]=pd.qcut(df[c],threshold,precision=0, labels=False, duplicates='drop')
    return df,discrete_df



if __name__ == '__main__':
    cvs_data='high_diamond_ranked_10min.csv'
    primitive_df=pd.read_csv(cvs_data)
    primitive_df=primitive_df.drop(columns='gameId')



    drop_features = ['blueGoldDiff', 'redGoldDiff',
                     'blueExperienceDiff', 'redExperienceDiff',
                     'blueCSPerMin', 'redCSPerMin',
                     'blueGoldPerMin', 'redGoldPerMin', 'blueFirstBlood', 'redFirstBlood']
    df=primitive_df.drop(columns=drop_features)
    special_features_names = [c[3:] for c in df.columns if c.startswith('red')]
    for name in special_features_names:
        df['br'+name]=df['blue'+name]-df['red'+name]
        df=df.drop(columns=['blue'+name,'red'+name])

    plt.figure(figsize=(18, 14))
    #display heatmap
    sns.heatmap(round(df.corr(), 2), cmap='Blues', annot=True)
    plt.show()

    # filtered useless features
    df = df.drop(columns=['brWardsDestroyed',
                          'brWardsPlaced', 'brTowersDestroyed', 'brTotalMinionsKilled', 'brHeralds'])
    discrete_df=df.copy()
    df,discrete_df=discrete_data_set(df,discrete_df)

    model_features=discrete_df.columns[1:]
    matrix=discrete_df[model_features].values
    labels=discrete_df['blueWins'].values
    train_matrix,test_matrix,train_labels,test_labels=train_test_split(matrix,labels,test_size=0.2, random_state=20030722)
    model= StrategyTree(classes=[0,1],features=model_features)
    model.fit(train_matrix,train_labels)

    result=model.predict(samples_matrix=test_matrix)
    print(result)































