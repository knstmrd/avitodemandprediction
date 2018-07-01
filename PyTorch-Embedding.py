# Pytorch with embeddings for categorical features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

pth = '/dev/null'

category_sizes = [47, 372, 3, 3064, 7, 278, 1277, 2402, 67, 46, 1824, 9]

X_train = sparse.load_npz(pth + 'sparse-features/train_full_sparse_noNA.npz')
Y = pd.read_csv(pth + 'sparse-features/train_item_ids_target.csv')
Y = Y['deal_probability'].values

ss = StandardScaler(with_mean=False)
X_train_noncat = ss.fit_transform(X_train[:, len(category_sizes):])
X_train = sparse.hstack((X_train[:, :len(category_sizes)], X_train_noncat), 'csc')

del X_train_noncat

X_train_cv, X_valid_cv, y_train_cv, y_valid_cv = train_test_split(X_train,
                                                                  Y,
                                                                  test_size=0.1, random_state=22)
del X_train, Y


def get_embed_size(n_categories):
    embed_size = (n_categories + 1) // 2 
    if embed_size > 50:
        return 50
    else:
        return embed_size


n_categories = len(category_sizes)
n_features_total = X_train_cv.shape[1]
n_features_noncat = X_train_cv.shape[1] - n_categories

total_embed_output_size = sum([get_embed_size(category_size) for category_size in category_sizes])


class FFNetEmbed(nn.Module):
    def __init__(self, h1, h2, d1, d2):
        super(FFNetEmbed, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(n_features_noncat +
                      total_embed_output_size, out_features=h1),
            nn.BatchNorm1d(num_features=h1),
            nn.ReLU(),
        )
        self.dropout1 = nn.Dropout(p=d1)
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=h1, out_features=h2),
            nn.BatchNorm1d(num_features=h2),
            nn.ReLU(),
        )
        self.dropout2 = nn.Dropout(p=d2)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=h2, out_features=1),
            nn.Sigmoid(),
        )

        self.embedding_layers = [nn.Embedding(category_size, get_embed_size(category_size))
                                 for category_size in category_sizes]

    def forward(self, x_cat, x_noncat):
        cat_through_embed = [self.embedding_layers[i](x_cat[:, i]) for i in range(n_categories)]

        x = torch.cat((*cat_through_embed, x_noncat), dim=1)

        x = self.input_layer(x)
        x = self.dropout1(x)
        x = self.hidden_layer1(x)
        x = self.dropout2(x)
        x = self.hidden_layer2(x)
        return x.view(-1)


def make_random_index():
    shuffle_index = np.arange(X_train_cv.shape[0])
    np.random.shuffle(shuffle_index)
    return shuffle_index


def get_batch_with_cat(no_batch, index):
    return (X_train_cv[index[no_batch * batchsize:batchsize * (no_batch + 1)], :12].todense(),
            X_train_cv[index[no_batch * batchsize:batchsize * (no_batch + 1)], 12:].todense(),
            y_train_cv[index[no_batch * batchsize:batchsize * (no_batch + 1)]])


logging_step = 200

epochs = 2
batchsize = 64
n_batches = int(X_train_cv.shape[0] / batchsize)
print(n_batches)

net_w_embed = FFNetEmbed(600, 400, 0.25, 0.25)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net_w_embed.parameters(), lr=1e-4, momentum=0.9)

for epoch in range(epochs):
    batch_idx = 0
    epoch_loss = 0.0

    print('Epoch', epoch, '/', epochs)
    index_arr = make_random_index()
    for batch_idx in range(n_batches):
        X_cat_b, X_noncat_b, y_b = get_batch_with_cat(batch_idx, index_arr)
        X_cat_b = X_cat_b.astype(int, copy=False)
        inputs_cat = torch.from_numpy(X_cat_b)
        inputs_noncat, labels = torch.from_numpy(X_noncat_b), torch.from_numpy(y_b)
        inputs_noncat = inputs_noncat.float()
        labels = labels.float()
        optimizer.zero_grad()
        outputs = net_w_embed(inputs_cat, inputs_noncat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.numpy()
        if batch_idx % logging_step == 0:
            
            rmse = 0
            for i in range(500):
                Xv = X_valid_cv[300 * i:300 * (i + 1), :].todense()
                Xv_c = Xv[:, :12].astype(int, copy=False)
                Xv_c = torch.from_numpy(Xv_c)
                Xv = torch.from_numpy(Xv[:, 12:])
                Xv = Xv.float()
                
                valid_out = net_w_embed(Xv_c, Xv)
                rmse += 300 * mean_squared_error(y_valid_cv[300 * i:300 * (i + 1)], valid_out.data.numpy())
            rmse = (rmse / 150000)**0.5
            print(batch_idx, epoch_loss / (batch_idx + 1),
                  'validation RMSE:', rmse)
