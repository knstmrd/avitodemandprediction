# main feature extraction code

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from re import findall
from lightgbm import plot_importance
from matplotlib import pyplot as plt
from datetime import datetime as dt
import string
import jamspell
from pymorphy2 import tokenizers
from re import sub
from nltk import tokenize


punct = set(string.punctuation)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


def compute_len(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_len'] = orig_df[col].apply(lambda x: len(x))
    return new_df


def encode_cols(df, col_names):
    encoders = {}
    for col in col_names:
        encoder = LabelEncoder()
        encoder.fit(df[col])
        encoders[col] = encoder
    return encoders


def count_chars_multiple(new_df, orig_df, chars, col_names):
    for col in col_names:
        start_count = False
        new_name = col + '_count_' + '_'.join(chars)
        for ch in chars:
            if start_count:
                new_df[new_name] += orig_df[col].apply(lambda x: x.count(ch))
            else:
                new_df[new_name] = orig_df[col].apply(lambda x: x.count(ch))
                start_count = True
    return new_df


def count_chars(new_df, orig_df, chars, col_names):
    for col in col_names:
        for ch in chars:
            new_df[col + '_count_' + ch] = orig_df[col].apply(lambda x: x.count(ch))
    return new_df


def compute_chars_digits(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_digits'] = orig_df[col].apply(lambda x: sum(c.isdigit() for c in x))
        new_df[col + '_chars'] = orig_df[col].apply(lambda x: sum(c.isalpha() for c in x))
    return new_df


def count_words(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_words'] = orig_df[col].apply(lambda x: len(x.split()))
    return new_df


def count_capital_letters(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_capital_letters'] = orig_df[col].apply(lambda x: sum(c.isupper() for c in x))
    return new_df


def count_bad_punctuation(new_df, orig_df, chars, col_names):
    for col in col_names:
        start_count = False
        new_name = col + '_count_badpc_' + '_'.join(chars)
        for ch in chars:
            rex = '\\' + ch + r'[^\s-]'
            if start_count:
                new_df[new_name] += orig_df[col].apply(lambda x: len(findall(rex, x)))
            else:
                new_df[new_name] = orig_df[col].apply(lambda x: len(findall(rex, x)))
                start_count = True
    return new_df


def count_bad_punctuation2(new_df, orig_df, chars, col_names):
    for col in col_names:
        start_count = False
        new_name = col + '_count_badpc2_' + '_'.join(chars)
        for ch in chars:
            rex = r'[^\s-]' + '\\' + ch
            if start_count:
                new_df[new_name] += orig_df[col].apply(lambda x: len(findall(rex, x)))
            else:
                new_df[new_name] = orig_df[col].apply(lambda x: len(findall(rex, x)))
                start_count = True
    return new_df


def count_english_chars(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_eng_chars'] = orig_df[col].apply(lambda x: len(findall('[a-zA-Z]', x)))
    return new_df


def count_unique_words(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_num_unique_words'] = orig_df[col].apply(lambda x: len(set(w.lower() for w in x.split())))
    return new_df


def compute_wl(x):
    splitstr = x.split()
    if len(splitstr) == 0:
        return 0
    else:
        return len(max(splitstr, key=len))


def longest_word(new_df, orig_df, col_names):
    for col in col_names:
        new_df[col + '_longest'] = orig_df[col].apply(lambda x: compute_wl(x))
    return new_df


def compute_percentage(df, column_names, divisor):
    for col_name in column_names:
        df[col_name + '_div_' + divisor] = df[col_name] / (df[divisor] + 1)
    return df


def count_multiple_exclamation(new_df, orig_df, col_names):
    rex = r'!{2,}'
    for col in col_names:
        new_name = col + '_multipleexclamation'
        new_df[new_name] = orig_df[col].apply(lambda x: len(findall(rex, x)))
    return new_df


def create_emoji_set(df_tr, df_te):
    emoji = set()
    for dataset in [df_tr, df_te]:
        for s in dataset['title'].fillna('').astype(str):
            for c in s:
                if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                    continue
                emoji.add(c)

        for s in dataset['description'].fillna('').astype(str):
            for c in str(s):
                if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                    continue
                emoji.add(c)
    return emoji


engineering_stage = 1

if engineering_stage == 1:
    for cn in ['param_1', 'param_2', 'param_3', 'title', 'description']:
        df_train[cn][df_train[cn].isnull()] = ''
        df_test[cn][df_test[cn].isnull()] = ''

    df_train['param_1'] = df_train['param_1'].apply(lambda x: str(x))
    df_train['param_2'] = df_train['param_2'].apply(lambda x: str(x))
    df_train['param_3'] = df_train['param_3'].apply(lambda x: str(x))
    df_train['title'] = df_train['title'].apply(lambda x: str(x))
    df_train['description'] = df_train['description'].apply(lambda x: str(x))
    df_train['has_param2'] = (~df_train['param_2'].isnull()).apply(lambda x: int(x))
    df_train['has_param3'] = (~df_train['param_3'].isnull()).apply(lambda x: int(x))

    df_test['param_1'] = df_test['param_1'].apply(lambda x: str(x))
    df_test['param_2'] = df_test['param_2'].apply(lambda x: str(x))
    df_test['param_3'] = df_test['param_3'].apply(lambda x: str(x))
    df_test['title'] = df_test['title'].apply(lambda x: str(x))
    df_test['description'] = df_test['description'].apply(lambda x: str(x))
    df_test['has_param2'] = (~df_test['param_2'].isnull()).apply(lambda x: int(x))
    df_test['has_param3'] = (~df_test['param_3'].isnull()).apply(lambda x: int(x))

    basic_encs = encode_cols(df_train, ['region', 'city', 'parent_category_name',
                                        'category_name', 'param_1', 'user_type'])

    double_enc_cols = ['param_2', 'param_3', 'city']

    d_e_df = pd.concat((df_train[double_enc_cols], df_test[double_enc_cols]))
    double_encs = encode_cols(d_e_df, double_enc_cols)

    train = pd.DataFrame()
    test = pd.DataFrame()
    
    train['item_id'] = df_train['item_id']
    test['item_id'] = df_test['item_id']

    train['text_feat'] = df_train.apply(lambda row: ' '.join([(row['param_1']),
                                                              (row['param_2']),
                                                              (row['param_3'])]), axis=1)
    test['text_feat'] = df_test.apply(lambda row: ' '.join([(row['param_1']),
                                                            (row['param_2']),
                                                            (row['param_3'])]), axis=1)

    text_feat_cols = ['text_feat']

    d_e_df = pd.concat((train[text_feat_cols], test[text_feat_cols]))
    text_feat_enc = encode_cols(d_e_df, text_feat_cols)

    encoded_cols_to_use = ['region', 'category_name', 'param_1', 'parent_category_name', 'user_type']

    for encol in encoded_cols_to_use:
        train[encol] = basic_encs[encol].transform(df_train[encol])
        test[encol] = basic_encs[encol].transform(df_test[encol])

    for encol in double_enc_cols:
        train[encol] = double_encs[encol].transform(df_train[encol])
        test[encol] = double_encs[encol].transform(df_test[encol])

    for encol in text_feat_cols:
        train[encol] = text_feat_enc[encol].transform(train[encol])
        test[encol] = text_feat_enc[encol].transform(test[encol])

    train = compute_len(train, df_train, ['title', 'description'])
    test = compute_len(test, df_test, ['title', 'description'])

    train = compute_chars_digits(train, df_train, ['title', 'description'])
    test = compute_chars_digits(test, df_test, ['title', 'description'])

    train = count_chars(train, df_train, ['!', '?', '$', '%'], ['title', 'description'])
    test = count_chars(test, df_test, ['!', '?', '$', '%'], ['title', 'description'])

    train = count_chars_multiple(train, df_train, ['@', '#', '^', '&', '*', '\\', '|', ':'],
                                 ['title', 'description'])
    test = count_chars_multiple(test, df_test, ['@', '#', '^', '&', '*', '\\', '|', ':'],
                                ['title', 'description'])

    train = count_chars_multiple(train, df_train, ['/', '<', '>', '=', '+'], ['title', 'description'])
    test = count_chars_multiple(test, df_test, ['/', '<', '>', '=', '+'], ['title', 'description'])

    train = count_unique_words(train, df_train, ['title', 'description'])
    test = count_unique_words(test, df_test, ['title', 'description'])

    train = count_words(train, df_train, ['title', 'description'])
    test = count_words(test, df_test, ['title', 'description'])

    train['price'] = df_train['price']
    test['price'] = df_test['price']

    train["price"] = np.log(train["price"] + 0.001)
    train["price"].fillna(-999, inplace=True)
    test["price"] = np.log(test["price"] + 0.001)
    test["price"].fillna(-999, inplace=True)

    train = count_capital_letters(train, df_train, ['title', 'description'])
    test = count_capital_letters(test, df_test, ['title', 'description'])

    train = count_bad_punctuation(train, df_train, ['.', ','], ['title', 'description'])
    test = count_bad_punctuation(test, df_test, ['.', ','], ['title', 'description'])

    train = count_chars(train, df_train, ['(', ')'], ['description'])
    test = count_chars(test, df_test, ['(', ')'], ['description'])

    train = count_bad_punctuation(train, df_train, [')'], ['title', 'description'])
    test = count_bad_punctuation(test, df_test, [')'], ['title', 'description'])

    train = count_bad_punctuation2(train, df_train, ['('], ['title', 'description'])
    test = count_bad_punctuation2(test, df_test, ['('], ['title', 'description'])

    train['unbalanced_desc_('] = abs(train['description_count_('] - train['description_count_)'])
    test['unbalanced_desc_('] = abs(test['description_count_('] - test['description_count_)'])

    train['image_top_1'] = df_train['image_top_1']
    test['image_top_1'] = df_test['image_top_1']

    train['image_top_1'][df_train['image'].isnull()] = -1
    test['image_top_1'][df_test['image'].isnull()] = -1

    train['item_seq_no'] = df_train['item_seq_number']
    test['item_seq_no'] = df_test['item_seq_number']

    train = count_english_chars(train, df_train, ['title', 'description'])
    test = count_english_chars(test, df_test, ['title', 'description'])

    df_train['activation_date'] = df_train['activation_date'].astype('datetime64[ns]')
    df_test['activation_date'] = df_test['activation_date'].astype('datetime64[ns]')
    train['weekday'] = df_train["activation_date"].dt.weekday
    test['weekday'] = df_test["activation_date"].dt.weekday

    train = longest_word(train, df_train, ['title', 'description'])
    test = longest_word(test, df_test, ['title', 'description'])
    
    for sch in ['!', '?', '%', '+', '=']:
        train = count_chars(train, df_train, [sch], ['title', 'description'])
        test = count_chars(test, df_test, [sch], ['title', 'description'])

        train = compute_percentage(train, ['description_count_' + sch], 'description_len')
        test = compute_percentage(test, ['description_count_' + sch], 'description_len')

        train = compute_percentage(train, ['title_count_' + sch], 'title_len')
        test = compute_percentage(test, ['title_count_' + sch], 'title_len')
        
    train = compute_percentage(train, ['title_capital_letters', 'title_eng_chars',
                                       'title_digits', 'title_words', 'title_longest'], 'title_len')
    test = compute_percentage(test, ['title_capital_letters', 'title_eng_chars',
                                     'title_digits', 'title_words', 'title_longest'], 'title_len')

    train = compute_percentage(train, ['description_capital_letters', 'description_eng_chars',
                                       'description_digits', 'description_count_/_<_>_=_+',
                                       'description_count_badpc_._,',
                                       'description_longest'], 'description_len')
    test = compute_percentage(test, ['description_capital_letters', 'description_eng_chars',
                                     'description_digits', 'description_count_/_<_>_=_+',
                                     'description_count_badpc_._,',
                                     'description_longest'], 'description_len')

    train = compute_percentage(train, ['description_len',
                                       'description_num_unique_words'], 'description_words')
    test = compute_percentage(test, ['description_len',
                                     'description_num_unique_words'], 'description_words')

    train = compute_percentage(train, ['title_len',
                                       'title_num_unique_words'], 'title_words')
    test = compute_percentage(test, ['title_len',
                                     'title_num_unique_words'], 'title_words')

    train.to_csv('train_v12.csv', index=False)
    test.to_csv('test_v12.csv', index=False)
    engineering_stage = 2


if engineering_stage == 2:
    train = pd.read_csv('train_v12.csv')
    test = pd.read_csv('test_v12.csv')
    
    df_train_norm_title = pd.read_csv('train_title_pym.csv')
    df_test_norm_title = pd.read_csv('test_title_pym.csv')
    
    df_train_norm_desc = pd.read_csv('train_descr_pym.csv')
    df_test_norm_desc = pd.read_csv('test_descr_pym.csv')
    
    base_pym_cols = ['norm_text', 'words_known', 'NOUN_count',
                     'ADJF_count', 'ADJS_count', 'COMP_count',
                     'VERB_count', 'INFN_count', 'PRTF_count',
                     'PRTS_count', 'GRND_count', 'NUMR_count',
                     'ADVB_count', 'NPRO_count', 'PRED_count',
                     'PREP_count', 'CONJ_count', 'PRCL_count',
                     'INTJ_count', 'misc_count']
    cc_title = {bpc: 'title_' + bpc for bpc in base_pym_cols}
    cc_desc = {bpc: 'description_' + bpc for bpc in base_pym_cols}
    
    df_train_norm_title.rename(columns=cc_title, inplace=True)
    df_test_norm_title.rename(columns=cc_title, inplace=True)
    print(df_test_norm_title.columns)
    df_train_norm_desc.rename(columns=cc_desc, inplace=True)
    df_test_norm_desc.rename(columns=cc_desc, inplace=True)
    print(df_test_norm_desc.columns)
    
    train = train.merge(df_train_norm_title, on='item_id')
    train = train.merge(df_train_norm_desc, on='item_id')
    test = test.merge(df_test_norm_title, on='item_id')
    test = test.merge(df_test_norm_desc, on='item_id')
    
    pos_cols = ['NOUN_count',
                'ADJF_count', 'ADJS_count', 'COMP_count',
                'VERB_count', 'INFN_count', 'PRTF_count',
                'PRTS_count', 'GRND_count', 'NUMR_count',
                'ADVB_count', 'NPRO_count', 'PRED_count',
                'PREP_count', 'CONJ_count', 'PRCL_count',
                'INTJ_count', 'misc_count']
    
    cc_title2 = ['title_' + bpc for bpc in pos_cols]
    cc_desc2 = ['description_' + bpc for bpc in pos_cols]
        
    train['title_norm_text'] = train['title_norm_text'].apply(lambda x: str(x))
    test['title_norm_text'] = test['title_norm_text'].apply(lambda x: str(x))
    train['description_norm_text'] = train['description_norm_text'].apply(lambda x: str(x))
    test['description_norm_text'] = test['description_norm_text'].apply(lambda x: str(x))
        
    train = count_unique_words(train, train, ['title_norm_text'])
    train = compute_percentage(train, cc_title2, 'title_words')

    test = count_unique_words(test, test, ['title_norm_text'])
    test = compute_percentage(test, cc_title2, 'title_words')
    
    train = count_unique_words(train, train, ['description_norm_text'])
    train = compute_percentage(train, cc_desc2, 'description_words')

    test = count_unique_words(test, test, ['description_norm_text'])
    test = compute_percentage(test, cc_desc2, 'description_words')
    
    print(train.columns, len(train.columns))  # 158 cols
    test.to_csv('test_v14v2.csv', index=False)
    train.to_csv('train_v14v2.csv', index=False)
    engineering_stage = 3


if engineering_stage == 3:
    train = pd.read_csv('train_v14v2.csv')
    test = pd.read_csv('test_v14v2.csv')
    
    df_train_svd5 = pd.read_csv('train_svd5_tfidf.csv')
    df_test_svd5 = pd.read_csv('test_svd5_tfidf.csv')
    
    print(df_train_svd5.columns)
    
    train = train.merge(df_train_svd5, on='item_id')
    test = test.merge(df_test_svd5, on='item_id')
    
    test.to_csv('test_v18v2.csv', index=False)
    train.to_csv('train_v18v2.csv', index=False)
    engineering_stage = 4


if engineering_stage == 4:
    train = pd.read_csv('train_v18v2.csv')
    test = pd.read_csv('test_v18v2.csv')
    
    df_train_svd5 = pd.read_csv('train_norm_svd5_tfidf.csv')
    df_test_svd5 = pd.read_csv('test_norm_svd5_tfidf.csv')
    
    print(df_train_svd5.columns)
    cols = df_train_svd5.columns
    
    new_cols = {ccc: 'pymorphy_' + ccc for ccc in cols if ccc != 'item_id'}
    df_train_svd5.rename(columns=new_cols, inplace=True)
    df_test_svd5.rename(columns=new_cols, inplace=True)
    
    train = train.merge(df_train_svd5, on='item_id')
    test = test.merge(df_test_svd5, on='item_id')
    
    test.to_csv('test_v18_5.csv', index=False)
    train.to_csv('train_v18_5.csv', index=False)
    engineering_stage = 5


if engineering_stage == 5:
    train = pd.read_csv('train_v18_5.csv')
    test = pd.read_csv('test_v18_5.csv')
    
    df_train_svd5 = pd.read_csv('train_norm_svd5_22_tfidf.csv')
    df_test_svd5 = pd.read_csv('test_norm_svd5_22_tfidf.csv')
    
    train = train.merge(df_train_svd5, on='item_id')
    test = test.merge(df_test_svd5, on='item_id')
    
    test.to_csv('test_v23.csv', index=False)
    train.to_csv('train_v23.csv', index=False)
    engineering_stage = 6


if engineering_stage == 6:
    df_train = pd.read_csv('train.csv',
                           usecols=['item_id', 'region', 'city'])
    df_test = pd.read_csv('test.csv',
                          usecols=['item_id', 'region', 'city'])
    df_geo = pd.read_csv('avito_region_city_features.csv')
    df_geo.drop(['city_region', 'region_id'], axis=1, inplace=True)
    df_train = df_train.merge(df_geo, how="left", on=["region", "city"])
    df_test = df_test.merge(df_geo, how="left", on=["region", "city"])
    df_train.drop(['region', 'city'], axis=1, inplace=True)
    df_test.drop(['region', 'city'], axis=1, inplace=True)
    train = pd.read_csv('train_v23.csv')
    test = pd.read_csv('test_v23.csv')
    
    train = train.merge(df_train, on='item_id')
    test = test.merge(df_test, on='item_id')

    print(train.shape, test.shape)
    test.to_csv('test_v28.csv', index=False)
    train.to_csv('train_v28.csv', index=False)
    engineering_stage = 7


if engineering_stage == 7:
    train = pd.read_csv('train_v28.csv')
    test = pd.read_csv('test_v28.csv')
    df_aggregated = pd.read_csv('aggregated_features.csv')
    
    df_train = pd.read_csv('train.csv',
                           usecols=('deal_probability', 'user_id', 'item_id'))
    df_test = pd.read_csv('test.csv',
                           usecols=('user_id', 'item_id'))
    train = train.merge(df_train, on='item_id', how='left')
    test = test.merge(df_test, on='item_id', how='left')

    train = train.merge(df_aggregated, on='user_id', how='left')
    test = test.merge(df_aggregated, on='user_id', how='left')
    test.to_csv('test_v32.csv', index=False)
    train.to_csv('train_v32.csv', index=False)
    engineering_stage = 8


if engineering_stage == 8:
    train = pd.read_csv('train_v32.csv')
    test = pd.read_csv('test_v32.csv')
    
    n_train = len(train)
    
    feat_cols_list_b = [t_cc for t_cc in train.columns if t_cc not in
                        ('title_norm_text', 'description_norm_text', 'deal_probability',
                         'item_id', 'Unnamed: 0', 'user_id', 'city', 'region',
                         'lat_lon_hdbscan_cluster_20_03', 'latitude', 'longitude')]

    categorical_f_b = ['region', 'category_name', 'param_1', 'user_type',
                       'image_top_1', 'weekday', 'param_2', 'param_3', 'city', 'text_feat',
                       'lat_lon_hdbscan_cluster_05_03',
                       'lat_lon_hdbscan_cluster_10_03', 'lat_lon_hdbscan_cluster_20_03',
                       'city_region_id']

    categorical_f = [cf for cf in categorical_f_b if cf in feat_cols_list_b]
    feat_cols_noncat = [fc for fc in feat_cols_list_b if fc not in categorical_f]
    
    df_total = pd.concat((train[categorical_f], test[categorical_f]))
    
    def check_onehot_ready(unique_vals):
        list_vals = unique_vals.tolist()
        list_vals.sort()
        is_ok = True
        for i in range(len(list_vals) - 1):
            if list_vals[i + 1] - list_vals[i] != 1:
                is_ok = False
            if list_vals[i] < 0:
                is_ok = False
        return is_ok

    fix_cols = []

    for cf_col in categorical_f:
        uv = df_total[cf_col].unique()
        ok_col = check_onehot_ready(uv)
        if not ok_col:
            fix_cols.append(cf_col)

    for bad_f_col in fix_cols:
        encoder = LabelEncoder()
        df_total[bad_f_col] = encoder.fit_transform(df_total[bad_f_col])
        train[bad_f_col] = df_total[bad_f_col][:n_train]
        test[bad_f_col] = df_total[bad_f_col][n_train:]
        
    test.to_csv('test_v33.csv', index=False)
    train.to_csv('train_v33.csv', index=False)
    engineering_stage = 9


if engineering_stage == 9:
    train = pd.read_csv('/Volumes/KM/Archive/KaggleStuff/2018Avito/train_v33.csv')
    test = pd.read_csv('/Volumes/KM/Archive/KaggleStuff/2018Avito/test_v33.csv')
    df_train = pd.read_csv('/Volumes/KM/Archive/KaggleStuff/2018Avito/train.csv',
                           usecols=['item_id', 'description', 'title'])
    df_test = pd.read_csv('/Volumes/KM/Archive/KaggleStuff/2018Avito/test.csv',
                          usecols=['item_id', 'description', 'title'])

    df_train['title'].fillna('', inplace=True)
    df_train['description'].fillna('', inplace=True)

    df_test['title'].fillna('', inplace=True)
    df_test['description'].fillna('', inplace=True)

    train = train.merge(df_train, on='item_id', how='left')
    test = test.merge(df_test, on='item_id', how='left')

    del df_train, df_test

    emj = create_emoji_set(train, test)

    train = count_multiple_exclamation(train, train, ['description'])
    test = count_multiple_exclamation(test, test, ['description'])

    train = compute_percentage(train, ['description_multipleexclamation'], 'description_len')
    test = compute_percentage(test, ['description_multipleexclamation'], 'description_len')

    train['n_title_emo'] = train['title'].apply(lambda x: sum(c in emj for c in x))
    train['n_description_emo'] = train['description'].apply(lambda x: sum(c in emj for c in x))

    test['n_title_emo'] = test['title'].apply(lambda x: sum(c in emj for c in x))
    test['n_description_emo'] = test['description'].apply(lambda x: sum(c in emj for c in x))

    train = compute_percentage(train, ['n_title_emo'], 'title_len')
    test = compute_percentage(test, ['n_title_emo'], 'title_len')
    train = compute_percentage(train, ['n_description_emo'], 'description_len')
    test = compute_percentage(test, ['n_description_emo'], 'description_len')

    train.drop(['title', 'description'], axis=1, inplace=True)
    test.drop(['title', 'description'], axis=1, inplace=True)

    test.to_csv('/Volumes/KM/Archive/KaggleStuff/2018Avito/test_v34.csv', index=False)
    train.to_csv('/Volumes/KM/Archive/KaggleStuff/2018Avito//train_v34.csv', index=False)
    engineering_stage = 10


if engineering_stage == 10:
    train = pd.read_csv('train_v34.csv')
    test = pd.read_csv('test_v34.csv')
    test.drop(['title_norm_text', 'description_norm_text'], axis=1, inplace=True)
    train.drop(['title_norm_text', 'description_norm_text'], axis=1, inplace=True)
    w2v_svd = np.load('desc_sum_w2v_fasttext_svd5.npy')
    n_train = len(train)
    for i in range(5):
        train['svd_desc_w2v_fasttext_' + str(i + 1)] = w2v_svd[:n_train, i]
        test['svd_desc_w2v_fasttext_' + str(i + 1)] = w2v_svd[n_train:, i]

    df_train_base = pd.read_csv('train.csv',
                                usecols=('item_id', 'image'))
    df_test_base = pd.read_csv('test.csv',
                               usecols=('item_id', 'image'))
    
    train_img = pd.read_csv('img_train_95c.csv')
    test_img = pd.read_csv('img_features_test.csv')
    
    to_drop = ['contrast', 'g_avg', 'g_skew', 'g_kurt', 'hu_mom_5', 'dist_centroid_avg']
    test_img.drop(to_drop, axis=1, inplace=True)
    if 'Unnamed: 0' in train_img.columns:
        train_img.drop(['Unnamed: 0'], axis=1, inplace=True)
        
    train_img.rename(columns={'images': 'image'}, inplace=True)
    test_img.rename(columns={'images': 'image'}, inplace=True)
    
    df_train_base = df_train_base.merge(train_img, on='image')
    df_test_base = df_test_base.merge(test_img, on='image')
    
    df_train_base.drop(['image'], axis=1, inplace=True)
    df_test_base.drop(['image'], axis=1, inplace=True)
    
    train = train.merge(df_train_base, on='item_id', how='left')
    test = test.merge(df_test_base, on='item_id', how='left')
    
    test.to_csv('test_v51.csv', index=False)
    train.to_csv('train_v51.csv', index=False)
    engineering_stage = 11


if engineering_stage == 11:
    train = pd.read_csv('train_v51.csv')
    test = pd.read_csv('test_v51.csv')

    w2v_svd = np.load('desc_sum_w2v_corpusnorm_svd5.npy')
    n_train = len(train)
    for i in range(5):
        train['svd_desc_w2v_normcorpus_' + str(i + 1)] = w2v_svd[:n_train, i]
        test['svd_desc_w2v_normcorpus_' + str(i + 1)] = w2v_svd[n_train:, i]

    test.to_hdf('features.h5', 'test_v59', table=True, mode='a')
    train.to_hdf('features.h5', 'train_v59', table=True, mode='a')

