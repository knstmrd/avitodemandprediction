# This is a simple averaging of OOF submissions
# starting with a step-size and then looking around the best weight combination with a smaller step

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import cartesian

pth = '/dev/null'
deal_prob = pd.read_csv(pth + 'sparse-features/train_item_ids_target.csv')

lasso001 = pd.read_csv('OOF/LassoRidge/full_Lasso001.csv')
ridge1 = pd.read_csv('OOF/LassoRidge/full_Ridge1.csv')
goss = pd.read_csv('OOF/LGBM-GOSS40/full_LGBM.csv')
lgbm32 = pd.read_csv('OOF/LGBM32/LGBM.csv')
lgbm70 = pd.read_csv('OOF/LGBM70/LGBM.csv')


def calc_rmse(weights, data):
    return (mean_squared_error(data['deal_probability'], data[cols] @ weights))**0.5


def make_single_pd(truth, predictions):
    for i, pred in enumerate(predictions):
        pred['deal_probability'].clip(0, 1, inplace=True)
        renamed_df = pred.rename(columns={'deal_probability': 'pred_' + str(i)})
        truth = truth.merge(renamed_df, on='item_id')
    return truth


dfs_to_avg = [lasso001, ridge1, goss, lgbm32, lgbm70]
names = ['lasso001', 'ridge (1)', 'goss', 'lgbm32', 'lgbm70']
n_dfs = len(dfs_to_avg)


def find_weights(df, max_depth, init_step=0.1):
    trmse_weights = []
    for i in range(max_depth):
        curr_step = init_step / (2.0 ** i)
        if i == 0: 
            tbase_weights = [np.arange(0., 1, init_step) for i in range(n_dfs - 1)]
        else:
            tbase_weights = [np.arange(max(0., trmse_weights[0][1][i] - curr_step),
                                       min(1., trmse_weights[0][1][i] + curr_step * 2),
                                       curr_step) for i in range(n_dfs - 1)]
        tcartesian_w = cartesian(tbase_weights)
        tsummed_weights = np.sum(tcartesian_w, axis=1)
        tcartesian_w = tcartesian_w[tsummed_weights <= 1.0, :]
        tsummed_weights = tsummed_weights[tsummed_weights <= 1.0]
        tsummed_weights = tsummed_weights.reshape(-1, 1)
        tcartesian_w = np.hstack((tcartesian_w, 1. - tsummed_weights))
        
        print('Current depth:', str(i) + ';', tcartesian_w.shape[0], 'weight combinations')
        for j in range(tcartesian_w.shape[0]):
            if j % 100 == 0 and j > 100:
                print(j)
            trmse = calc_rmse(tcartesian_w[j, :], full_df)
            trmse_weights.append((trmse, tcartesian_w[j, :]))
            trmse_weights.sort(key=lambda x: x[0])
        print('Best result:', trmse_weights[0][0], '\n')
    return trmse_weights


full_df = make_single_pd(deal_prob, dfs_to_avg)
cols = [cc for cc in full_df.columns if cc not in ['deal_probability', 'item_id']]


rmse_weights = []
pure_rmse_weight = np.eye(n_dfs)
for i in range(n_dfs):
    print(names[i], calc_rmse(pure_rmse_weight[i, :], full_df))


out = find_weights(full_df, 10, 0.2)


print('RMSE:', out[0][0], '\n', list(zip(names, out[0][1])))

step_size = 0.0

base_weights = [np.arange(0., 1, step_size) for i in range(n_dfs - 1)]
cartesian_w = cartesian(base_weights)
summed_weights = np.sum(cartesian_w, axis=1)
cartesian_w = cartesian_w[summed_weights <= 1.0, :]
summed_weights = summed_weights[summed_weights <= 1.0]
summed_weights = summed_weights.reshape(-1, 1)
cartesian_w = np.hstack((cartesian_w, 1. - summed_weights))

print(cartesian_w.shape[0], 'weight combinations')
for i in range(cartesian_w.shape[0]):
    if i % 100 == 0:
        print(i)
    rmse = calc_rmse(cartesian_w[i, :], full_df)
    rmse_weights.append((rmse, cartesian_w[i, :]))

rmse_weights.sort(key=lambda x: x[0])
print(rmse_weights[:10])