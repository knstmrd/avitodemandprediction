# use pymorphy to normalize text and count parts of speech

import pandas as pd
from functools import lru_cache
import pymorphy2
from pymorphy2 import tokenizers
import re


LEMMATIZER = pymorphy2.MorphAnalyzer()

gram = ['NOUN',
        'ADJF',
        'ADJS',
        'COMP',
        'VERB',
        'INFN',
        'PRTF',
        'PRTS',
        'GRND',
        'NUMR',
        'ADVB',
        'NPRO',
        'PRED',
        'PREP',
        'CONJ',
        'PRCL',
        'INTJ',
        'misc']


@lru_cache(maxsize=100000)
def lemmatize(word):
    p = LEMMATIZER.parse(word)
    if p[0].tag.POS is None:
        return ('', 'misc')
    else:
        return (p[0].normal_form, str(p[0].tag.POS))


@lru_cache(maxsize=100000)
def word_is_known(word):
    return int(LEMMATIZER.word_is_known(word))


def lemmatize_text(text):
    text = re.sub("\s+", " ", text)
    split_text = tokenizers.simple_word_tokenize(text)
    lemmatized_text = [lemmatize(t) for t in split_text]
    lemmas = [l[0] for l in lemmatized_text]
    t_tags = [l[1] for l in lemmatized_text]
    tags = [t_tags.count(gm) for gm in gram]
    words_known = sum([word_is_known(t) for t in split_text])
    return ' '.join(lemmas), words_known, tags


# process titles

df_train = pd.read_csv('train.csv')
df_train = df_train[['item_id', 'title']]
df_train['title'][df_train['title'].isnull()] = ''
df_train['title'] = df_train['title'].apply(lambda x: str(x))

df_train['tmp_title_processed'] = df_train['title'].apply(lambda x: lemmatize_text(x))
df_train['title_norm_text'] = df_train['tmp_title_processed'].apply(lambda x: x[0])
df_train['title_words_known'] = df_train['tmp_title_processed'].apply(lambda x: x[1])

for i, col in enumerate(gram):
    df_train['title_' + col + '_count'] = df_train['tmp_title_processed'].apply(lambda x: x[2][i])

df_train.drop(['title', 'tmp_title_processed'], axis=1, inplace=True)
df_train.to_csv('train_title_pym.csv', index=False)


df_test = pd.read_csv('test.csv')
df_test = df_test[['item_id', 'title']]
df_test['title'][df_test['title'].isnull()] = ''
df_test['title'] = df_test['title'].apply(lambda x: str(x))

df_test['tmp_title_processed'] = df_test['title'].apply(lambda x: lemmatize_text(x))
df_test['title_norm_text'] = df_test['tmp_title_processed'].apply(lambda x: x[0])
df_test['title_words_known'] = df_test['tmp_title_processed'].apply(lambda x: x[1])

for i, col in enumerate(gram):
    df_test['title_' + col + '_count'] = df_test['tmp_title_processed'].apply(lambda x: x[2][i])

df_test.drop(['title', 'tmp_title_processed'], axis=1, inplace=True)
df_test.to_csv('test_title_pym.csv', index=False)


# process descriptions

df_train = pd.read_csv('train.csv')
df_train = df_train[['item_id', 'description']]
df_train['description'][df_train['description'].isnull()] = ''
df_train['description'] = df_train['description'].apply(lambda x: str(x))


df_train['tmp_description_processed'] = df_train['description'].apply(lambda x: lemmatize_text(x))
df_train['description_norm_text'] = df_train['tmp_description_processed'].apply(lambda x: x[0])
df_train['description_words_known'] = df_train['tmp_description_processed'].apply(lambda x: x[1])

for i, col in enumerate(gram):
    df_train['description_' + col + '_count'] = df_train['tmp_description_processed'].apply(lambda x: x[2][i])

df_train.drop(['description', 'tmp_description_processed'], axis=1, inplace=True)
df_train.to_csv('train_descr_pym.csv', index=False)


df_test = pd.read_csv('test.csv')
df_test = df_test[['item_id', 'description']]
df_test['description'][df_test['description'].isnull()] = ''
df_test['description'] = df_test['description'].apply(lambda x: str(x))


df_test['tmp_description_processed'] = df_test['description'].apply(lambda x: lemmatize_text(x))
df_test['description_norm_text'] = df_test['tmp_description_processed'].apply(lambda x: x[0])
df_test['description_words_known'] = df_test['tmp_description_processed'].apply(lambda x: x[1])

for i, col in enumerate(gram):
    df_test['description_' + col + '_count'] = df_test['tmp_description_processed'].apply(lambda x: x[2][i])

df_test.drop(['description', 'tmp_description_processed'], axis=1, inplace=True)
df_test.to_csv('test_descr_pym.csv', index=False)
