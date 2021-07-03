import pandas as pd
import numpy as np
import simplejson as js
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, plot_confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE

def model_eval(y_true, X_test, model, plot_matrix = False, scoring_strategy='macro'):
    y_pred = model.predict(X_test)
    p = precision_score(y_true, y_pred,average=scoring_strategy)
    r = recall_score(y_true, y_pred,average=scoring_strategy)
    f1 = f1_score(y_true, y_pred,average=scoring_strategy)
    a = accuracy_score(y_true, y_pred)
    b_a = balanced_accuracy_score(y_true, y_pred)
    if plot_matrix:
        f, ax = plt.subplots(2,1, figsize=(5,8))
        plot_confusion_matrix(model, X_test, y_true, ax = ax[0],cmap = plt.cm.Blues)
        plot_confusion_matrix(model, X_test, y_true, ax = ax[1], cmap = plt.cm.Blues, normalize='all')
    print('Precision = {:.2f}, Recall = {:.2f}, F1 = {:.2f}, Acc. = {:.2f}, Bal Acc. = {:.2f}'.format(p,r,f1,a,b_a))
    return p,r,f1,a, b_a

def gs_df_plot(gs, save_json=None, save_model=None):
    '''

    :param gs:
    :return:
    '''
    df = pd.DataFrame(gs.cv_results_)
    df.sort_values(by='rank_test_f1_macro',inplace=True)
    df_mod = df.copy(deep=True)
    for par in df_mod.params[0].keys():
        df_mod[par] = ''
    for idx, d in df_mod.iterrows():
        for par in df_mod.params[0].keys():
            try:
                df_mod.loc[idx, par] = d['params'][par]
            except:
                df_mod.loc[idx, par] = str(d['params'][par])
    cols = ['rank_test_f1_macro'] + list(df.params[0].keys()) + \
           ['mean_test_f1_macro','mean_test_accuracy',
            'std_test_f1_macro','std_test_accuracy',
            'mean_train_f1_macro','mean_train_accuracy',
            'std_train_f1_macro','std_train_accuracy']
    data = df_mod.loc[:, cols]
    data.sort_values(by='rank_test_f1_macro',inplace=True)
    data.reset_index(drop=True,inplace=True)

    f, ax = plot_gs_res(data)

    if save_json is not None:
        data.reset_index(drop=True).to_json(save_json, indent=4, orient='index')

    if save_model is not None:
        joblib.dump(gs,save_model)

    return data, df, f

def plot_gs_res(data):
    # Plot section
    f, ax = plt.subplots(2, 1, sharex=True)
    C0 = 'C0'
    C1 = 'C1'
    try:
        data.plot(y=['mean_train_f1_macro', 'mean_test_f1_macro'], style='o-', color=[C0, C1], grid=True, ax=ax[0])
        data.plot(y=['mean_train_accuracy', 'mean_test_accuracy'], style='o-', color=[C0, C1], grid=True, ax=ax[1])
    except:
        data.plot(y=['mean_train_f1_macro', 'mean_test_f1_macro'], style='o-', grid=True, ax=ax[0])
        data.plot(y=['mean_train_accuracy', 'mean_test_accuracy'], style='o-', grid=True, ax=ax[1])

    data['ub_train_f1'] = data['mean_train_f1_macro'] + data['std_train_f1_macro']
    data['lb_train_f1'] = data['mean_train_f1_macro'] - data['std_train_f1_macro']
    data['ub_train_acc'] = data['mean_train_accuracy'] + data['std_train_accuracy']
    data['lb_train_acc'] = data['mean_train_accuracy'] - data['std_train_accuracy']
    data['ub_test_f1'] = data['mean_test_f1_macro'] + data['std_test_f1_macro']
    data['lb_test_f1'] = data['mean_test_f1_macro'] - data['std_test_f1_macro']
    data['ub_test_acc'] = data['mean_test_accuracy'] + data['std_test_accuracy']
    data['lb_test_acc'] = data['mean_test_accuracy'] - data['std_test_accuracy']
    try:
        ax[0].fill_between(data.index.values,
                       data['lb_train_f1'], data['ub_train_f1'], alpha=0.2,
                       color=C0, lw= 2)
        ax[0].fill_between(data.index.values,
                       data['lb_test_f1'], data['ub_test_f1'], alpha=0.2,
                       color=C1, lw= 2)
        ax[1].fill_between(data.index.values,
                       data['lb_train_acc'], data['ub_train_acc'], alpha=0.2,
                       color=C0, lw= 2)
        ax[1].fill_between(data.index.values,
                       data['lb_test_acc'], data['ub_test_acc'], alpha=0.2,
                       color=C1, lw= 2)
    except:
        ax[0].fill_between(data.index.values,
                       data['lb_train_f1'], data['ub_train_f1'], alpha=0.2, lw= 2)
        ax[0].fill_between(data.index.values,
                       data['lb_test_f1'], data['ub_test_f1'], alpha=0.2, lw= 2)
        ax[1].fill_between(data.index.values,
                       data['lb_train_acc'], data['ub_train_acc'], alpha=0.2, lw= 2)
        ax[1].fill_between(data.index.values,
                       data['lb_test_acc'], data['ub_test_acc'], alpha=0.2, lw= 2)
    return f, ax