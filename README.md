# Avito Demand Prediction Challenge

Some code snippets used for the [Kaggle Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction).
End result: top 28% on private LB, 12% on public LB.
Code is a bit of a mess due to small amount of time spent on actually cleaning it up.

End model is a LGBM that is trained on OOF predictions of several LGBM/XGB/Catboost/Linear Regression models.

While there is some PyTorch code in this repo, the model was slow to train on a CPU, and I haven't tried renting a GPU instance somewhere.

# Features

Apart from some features already present in the dataset (like price), several different kinds of features were used:

1. Various statistics - counts of punctuation characters, emojis, bad punctuation, unique words, digits, lengths of words, etc.
1. 300-dimensional w2vec embeddings of title and description (summ of word embeddings for all words); using pretrained embeddings (on Russian wiki corpus) and self-trained embeddings on the dataset in question. Since these arrays take up a large amount of RAM, 5 top SVD components were used as features
1. Image features (blurness, Hu moments, dullness, etc.), some ideas taken from [a kernel](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality) by [sban](https://www.kaggle.com/shivamb), some taken from the paper [Cheng, Haibin, et al. "Multimedia features for click prediction of new ads in display advertising." Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2012](https://www.researchgate.net/publication/229224063_Multimedia_Features_for_Click_Prediction_of_New_Ads_in_NGD_Display_Advertising). Extracted using OpenCV.
1. [Aggregated features](https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm), by [Benjamin Minixhofer](https://www.kaggle.com/bminixhofer)
1. [Geo features](https://www.kaggle.com/frankherfert/region-and-city-details-with-lat-lon-and-clusters), by [Frank Herfert](FrankHerfert)
