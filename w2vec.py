# W2V using pre-trained Russian wiki corpus and new corpus trained on Avito data

import numpy as np
import pandas as pd
import nltk
from gensim.models import KeyedVectors, word2vec
from sklearn.decomposition import TruncatedSVD


df_train = pd.read_csv('train_v34.csv',
                       usecols=('item_id', 'description_norm_text', 'title_norm_text'))
df_test = pd.read_csv('test_v34.csv',
                      usecols=('item_id', 'description_norm_text', 'title_norm_text'))


def tokenize(x):
    tok = nltk.tokenize.toktok.ToktokTokenizer()
    return [t.lower() for t in tok.tokenize(x)]


def get_vector(x):
    # If the word is out of vocab, then return a 300 dim vector filled with zeros
    try:
        return model.get_vector(x)
    except:
        return np.zeros(shape=300)


def vector_sum(x):
    return np.sum(x, axis=0)


# russian wiki corpus w2vec

model = KeyedVectors.load_word2vec_format('Data/wiki.ru.vec')

features_train_d = np.zeros((len(df_train), 300))
for i, desc in enumerate(df_train['description_norm_text'].values):
    tokens = tokenize(desc)
    if len(tokens) != 0:  # If the description is missing then return a 300 dim vector filled with zeros
        word_vecs = [get_vector(w) for w in tokens]
        features_train_d[i, :] = vector_sum(word_vecs)

features_test_d = np.zeros((len(df_test), 300))
for i, desc in enumerate(df_test['description_norm_text'].values):
    tokens = tokenize(desc)
    if len(tokens) != 0:  # If the description is missing then return a 300 dim vector filled with zeros
        word_vecs = [get_vector(w) for w in tokens]
        features_test_d[i, :] = vector_sum(word_vecs)

np.save("desc_sum_w2v_fasttext_train.npy", features_train_d)
np.save("desc_sum_w2v_fasttext_test.npy", features_test_d)

w2v_norm = np.load('desc_sum_w2v_fasttext_train.npy')
w2v_normt = np.load('desc_sum_w2v_fasttext_test.npy')

n_train = w2v_norm.shape[0]

overall = np.vstack((w2v_norm, w2v_normt))
del w2v_norm, w2v_normt

svd_w2v = TruncatedSVD(n_components=5)
svd_result = svd_w2v.fit_transform(overall)

print(svd_result.shape)
np.save("desc_sum_w2v_fasttext_svd5.npy", svd_result)

total_text = pd.concat([df_train['description_norm_text'],
                        df_train['title_norm_text'],
                        df_test['description_norm_text'],
                        df_test['title_norm_text']],
                       ignore_index=True)
total_text.fillna('', inplace=True)
total_text = total_text.apply(lambda x: x.split())

model = word2vec.Word2Vec(total_text, size=300, window=5, min_count=5, workers=4)

model = model.wv

features_train_d = np.zeros((len(df_train), 300))
for i, desc in enumerate(df_train['description_norm_text'].values):
    tokens = tokenize(desc)
    if len(tokens) != 0:  # If the description is missing then return a 300 dim vector filled with zeros
        word_vecs = [get_vector(w) for w in tokens]
        features_train_d[i, :] = vector_sum(word_vecs)
        
features_test_d = np.zeros((len(df_test), 300))
for i, desc in enumerate(df_test['description_norm_text'].values):
    tokens = tokenize(desc)
    if len(tokens) != 0:  # If the description is missing then return a 300 dim vector filled with zeros
        word_vecs = [get_vector(w) for w in tokens]
        features_test_d[i, :] = vector_sum(word_vecs)

np.save("desc_sum_w2v_corpusnorm_train.npy",
        features_train_d)
np.save("desc_sum_w2v_corpusnorm_test.npy",
        features_test_d)

n_train = len(df_train)

overall = np.vstack((features_train_d, features_test_d))

svd_w2v = TruncatedSVD(n_components=5)
svd_result = svd_w2v.fit_transform(overall)

print(svd_result.shape)
np.save("desc_sum_w2v_corpusnorm_svd5.npy", svd_result)
