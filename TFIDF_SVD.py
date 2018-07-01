# compute TF-IDF (and SVD components)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords


russian_stop = set(stopwords.words('russian'))

# this is extraction of TFIDF features and 5 SVD components from the non-normalized text

df_train = pd.read_csv('train.csv',
                       usecols=('item_id', 'description', 'title'))
df_test = pd.read_csv('test.csv',
                      usecols=('item_id', 'description', 'title'))

title_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.99)
desc_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.99)

for df in (df_train, df_test):
    for cc in ('description', 'title'):
        df[cc][df[cc].isnull()] = ''
        df[cc] = df[cc].apply(lambda x: str(x))

n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_title = title_vectorizer.fit_transform(df_total['title'])
print(X_title.shape)
sparse.save_npz("'title_tfidf.npz", X_title)

X_desc = desc_vectorizer.fit_transform(df_total['description'])
print(X_desc.shape)
sparse.save_npz("'desc_tfidf.npz", X_desc)

n_comp = 5
svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
title_comps = svd_title.fit_transform(X_title)

n_comp = 5
svd_desc = TruncatedSVD(n_components=n_comp, algorithm='arpack')
desc_comps = svd_desc.fit_transform(X_desc)

svd_train = pd.DataFrame()
svd_test = pd.DataFrame()
svd_train['item_id'] = df_train['item_id']
svd_test['item_id'] = df_test['item_id']

for i in range(n_comp):
    svd_train['svd_title_' + str(i + 1)] = title_comps[:n_train, i]
    svd_test['svd_title_' + str(i + 1)] = title_comps[n_train:, i]
    
    svd_train['svd_desc_' + str(i + 1)] = desc_comps[:n_train, i]
    svd_test['svd_desc_' + str(i + 1)] = desc_comps[n_train:, i]
    
svd_train.to_csv('train_svd5_tfidf.csv', index=False)
svd_test.to_csv('test_svd5_tfidf.csv', index=False)


# this is extraction of TFIDF features and 5 SVD components from the normalized text

df_train = pd.read_csv('train_v18.csv',
                       usecols=('item_id', 'description_norm_text', 'title_norm_text'))
df_test = pd.read_csv('test_v18.csv',
                       usecols=('item_id', 'description_norm_text', 'title_norm_text'))
title_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.99)
desc_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.99)

for df in (df_train, df_test):
    for cc in ('description_norm_text', 'title_norm_text'):
        df[cc][df[cc].isnull()] = ''
        df[cc] = df[cc].apply(lambda x: str(x))
        
n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_title = title_vectorizer.fit_transform(df_total['title_norm_text'])
print(X_title.shape)
sparse.save_npz("title_norm_tfidf.npz", X_title)

X_desc = desc_vectorizer.fit_transform(df_total['description_norm_text'])
print(X_desc.shape)
sparse.save_npz("desc_norm_tfidf.npz", X_desc)

n_comp = 5
svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
title_comps = svd_title.fit_transform(X_title)

n_comp = 5
svd_desc = TruncatedSVD(n_components=n_comp, algorithm='arpack')
desc_comps = svd_desc.fit_transform(X_desc)

svd_train = pd.DataFrame()
svd_test = pd.DataFrame()
svd_train['item_id'] = df_train['item_id']
svd_test['item_id'] = df_test['item_id']

for i in range(n_comp):
    svd_train['pymorphy_svd_title_' + str(i + 1)] = title_comps[:n_train, i]
    svd_test['pymorphy_svd_title_' + str(i + 1)] = title_comps[n_train:, i]
    
    svd_train['pymorphy_svd_desc_' + str(i + 1)] = desc_comps[:n_train, i]
    svd_test['pymorphy_svd_desc_' + str(i + 1)] = desc_comps[n_train:, i]
    
svd_train.to_csv('train_norm_svd5_tfidf.csv', index=False)
svd_test.to_csv('test_norm_svd5_tfidf.csv', index=False)


# In[8]:


# this is extraction of TFIDF features and 5 SVD components from the normalized description text
# using 2 grams



df_train = pd.read_csv('train_v18.csv',
                       usecols=('item_id', 'description_norm_text'))
df_test = pd.read_csv('test_v18.csv',
                       usecols=('item_id', 'description_norm_text'))
# title_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.99)
desc_vectorizer = TfidfVectorizer(stop_words=russian_stop,
                                  ngram_range=(2, 2),
                                  max_features=15000,
                                  sublinear_tf=True,
                                  lowercase=True,
                                  min_df=5, max_df=0.99)

for df in (df_train, df_test):
    df['description_norm_text'][df['description_norm_text'].isnull()] = ''
    df['description_norm_text'] = df['description_norm_text'].apply(lambda x: str(x))
        
n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_desc = desc_vectorizer.fit_transform(df_total['description_norm_text'])
print(X_desc.shape)
sparse.save_npz("desc_norm_22_tfidf.npz", X_desc)

n_comp = 5
svd_desc = TruncatedSVD(n_components=n_comp, algorithm='arpack')
desc_comps = svd_desc.fit_transform(X_desc)

svd_train = pd.DataFrame()
svd_test = pd.DataFrame()
svd_train['item_id'] = df_train['item_id']
svd_test['item_id'] = df_test['item_id']

for i in range(n_comp):
    svd_train['pymorphy_svd_desc_22_' + str(i+1)] = desc_comps[:n_train, i]
    svd_test['pymorphy_svd_desc_22_' + str(i+1)] = desc_comps[n_train:, i]
    
svd_train.to_csv('train_norm_svd5_22_tfidf.csv', index=False)
svd_test.to_csv('test_norm_svd5_22_tfidf.csv', index=False)


# In[5]:


# this is extraction of TFIDF features from the non-normalized text, with stop-words, and max_features

df_train = pd.read_csv('train.csv',
                       usecols=('item_id','description','title'))
df_test = pd.read_csv('test.csv',
                       usecols=('item_id','description','title'))

title_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=russian_stop,
                                   sublinear_tf=True,
                                   min_df=5, max_df=0.99,
                                   max_features=10000)
desc_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=russian_stop,
                                   sublinear_tf=True,
                                   min_df=5, max_df=0.99,
                                   max_features=15000)

for df in (df_train, df_test):
    for cc in ('description', 'title'):
        df[cc][df[cc].isnull()] = ''
        df[cc] = df[cc].apply(lambda x: str(x))
        
n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_title = title_vectorizer.fit_transform(df_total['title'])
print(X_title.shape)
sparse.save_npz("title_tfidfv2.npz", X_title)

X_desc = desc_vectorizer.fit_transform(df_total['description'])
print(X_desc.shape)
sparse.save_npz("desc_tfidfv2.npz", X_desc)


# In[3]:


# this is extraction of TFIDF features from the non-normalized text, with stop-words, and max_features

df_train = pd.read_csv('train.csv',
                       usecols=('item_id', 'description', 'title'))
df_test = pd.read_csv('test.csv',
                      usecols=('item_id', 'description', 'title'))

title_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=russian_stop,
                                   max_features=10000)
desc_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=russian_stop,
                                  max_features=15000)

for df in (df_train, df_test):
    for cc in ('description', 'title'):
        df[cc][df[cc].isnull()] = ''
        df[cc] = df[cc].apply(lambda x: str(x))
        
n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_title = title_vectorizer.fit_transform(df_total['title'])
print(X_title.shape)
sparse.save_npz("title_tfidfv3.npz", X_title)

X_desc = desc_vectorizer.fit_transform(df_total['description'])
print(X_desc.shape)
sparse.save_npz("desc_tfidfv3.npz", X_desc)


# In[4]:


df_train = pd.read_csv('train_v18.csv',
                       usecols=('item_id', 'description_norm_text'))
df_test = pd.read_csv('test_v18.csv',
                       usecols=('item_id', 'description_norm_text'))

desc_vectorizer = TfidfVectorizer(stop_words=russian_stop,
                                  ngram_range=(2, 2),
                                  max_features=15000)

for df in (df_train, df_test):
    df['description_norm_text'][df['description_norm_text'].isnull()] = ''
    df['description_norm_text'] = df['description_norm_text'].apply(lambda x: str(x))
        
n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_desc = desc_vectorizer.fit_transform(df_total['description_norm_text'])
print(X_desc.shape)
sparse.save_npz("desc_norm_22_tfidfv2.npz", X_desc)


# In[4]:


# this is extraction of TFIDF features from the normalized text, with stop-words, and max_features

df_train = pd.read_csv('train_v18.csv',
                       usecols=('item_id', 'description_norm_text', 'title_norm_text'))
df_test = pd.read_csv('test_v18.csv',
                      usecols=('item_id', 'description_norm_text', 'title_norm_text'))

title_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=russian_stop,
                                   sublinear_tf=True,
                                   min_df=5, max_df=0.99,
                                   max_features=10000)
desc_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=russian_stop,
                                  sublinear_tf=True,
                                  min_df=5, max_df=0.99,
                                  max_features=15000)

for df in (df_train, df_test):
    for cc in ('description_norm_text', 'title_norm_text'):
        df[cc][df[cc].isnull()] = ''
        df[cc] = df[cc].apply(lambda x: str(x))
        
n_train = len(df_train)
n_test = len(df_test)

df_total = pd.concat((df_train, df_test))

X_title = title_vectorizer.fit_transform(df_total['title_norm_text'])
print(X_title.shape)
sparse.save_npz("title_norm_tfidfv2.npz", X_title)

X_desc = desc_vectorizer.fit_transform(df_total['description_norm_text'])
print(X_desc.shape)
sparse.save_npz("desc_norm_tfidfv2.npz", X_desc)
