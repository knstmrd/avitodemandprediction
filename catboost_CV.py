# 4 folds of OOF and a submission using Catboost by Yandex

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold
import csv
from catboost import CatBoostRegressor

pth = '../'
out_pth = '../OOF/Catboost6000/'

with open(pth + 'sparse-features/fnames.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        fnames = row

deal_prob = pd.read_csv(pth + 'sparse-features/train_item_ids_target.csv')
train_arr = sparse.load_npz(pth + 'sparse-features/train_full_sparse.npz')
train_arr = train_arr[:, :244].todense()  # the other features are the large TF-IDF part, which doesn't fit in RAM
train_arr = np.array(train_arr)
Y = deal_prob['deal_probability'].values


kf = KFold(n_splits=5, random_state=14)
n_folds_max = 4


depth = 7
lr = 0.03
onehotmax = 3
metric_period = 200
colsample_bylevel = 0.4
niter = 6000

counter = 0
for train_index, test_index in kf.split(train_arr):
    if counter < n_folds_max:
        counter += 1
        reg_CB = CatBoostRegressor(learning_rate=lr, iterations=niter,
        						   one_hot_max_size=onehotmax,
        						   depth=depth,
        						   colsample_bylevel=0.4,
        						   metric_period=metric_period)

        print('Training CB, fold', counter)
        reg_CB.fit(train_arr[train_index, :], Y[train_index],
                   cat_features=list(range(12)))

        print('Predicting')
        pred_CB = reg_CB.predict(train_arr[test_index, :])

        item_ids = deal_prob['item_id'][test_index]
        pd_CB = {'item_id': item_ids, 'deal_probability': pred_CB}
        pd_CB = pd.DataFrame(pd_CB)
        pd_CB.to_csv(out_pth + 'CB_fold_' + str(counter) + '.csv', index=False)


reg_CB = CatBoostRegressor(learning_rate=lr, iterations=niter,
						   one_hot_max_size=onehotmax,
						   depth=depth,
						   colsample_bylevel=0.4,
        				   metric_period=metric_period)

print('Training on full train set')
reg_CB.fit(train_arr, Y)
del train_arr

test_arr = sparse.load_npz(pth + 'sparse-features/test_full_sparse.npz')
test_item_ids = pd.read_csv(pth + 'sparse-features/test_item_ids.csv')
test_arr = test_arr[:, :244].todense()
test_arr = np.array(test_arr)

output = reg_CB.predict(test_arr)
output[output<0] = 0
output[output>1] = 1
sub = pd.DataFrame({'item_id': test_item_ids['item_id'], 'deal_probability': output})
sub.to_csv('submissions/CB6k.csv', index=False, header=True)
