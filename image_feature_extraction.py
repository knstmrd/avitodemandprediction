# openCV image feature extraction

import cv2
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from time import perf_counter
from tqdm import tqdm


def getSize(filename):
    st = os.stat(filename)
    return st.st_size


def getDimensions(img):
    return (img.size, img.shape[0] / img.shape[1])


def get_blurrness_score(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def get_rgb_features(img):
    min_r = np.min(img[:, :, 0])
    max_r = np.max(img[:, :, 0])
    std_r = np.std(img[:, :, 0])
    avg_r = np.mean(img[:, :, 0])
    skew_r = skew(img[:, :, 0], axis=None)
    kurt_r = kurtosis(img[:, :, 0], axis=None)
    
    min_g = np.min(img[:, :, 1])
    max_g = np.max(img[:, :, 1])
    std_g = np.std(img[:, :, 1])
    avg_g = np.mean(img[:, :, 1])
    skew_g = skew(img[:, :, 1], axis=None)
    kurt_g = kurtosis(img[:, :, 1], axis=None)
    
    min_b = np.min(img[:, :, 2])
    max_b = np.max(img[:, :, 2])
    std_b = np.std(img[:, :, 2])
    avg_b = np.mean(img[:, :, 2])
    skew_b = skew(img[:, :, 2], axis=None)
    kurt_b = kurtosis(img[:, :, 2], axis=None)
    
    return [min_r, max_r, std_r, avg_r, skew_r, kurt_r,
            min_g, max_g, std_g, avg_g, skew_g, kurt_g,
            min_b, max_b, std_b, avg_b, skew_b, kurt_b]


def get_hsv_features(img_scaled):
    #OK, working
    hls = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2HLS)
    min_l = np.min(hls[:, :, 1])
    max_l = np.max(hls[:, :, 1])
    std_l = np.std(hls[:, :, 1])
    avg_l = np.mean(hls[:, :, 1])
    skew_l = skew(hls[:, :, 1], axis=None)
    kurt_l = kurtosis(hls[:, :, 1], axis=None)
    
    min_s = np.min(hls[:, :, 2])
    max_s = np.max(hls[:, :, 2])
    std_s = np.std(hls[:, :, 2])
    avg_s = np.mean(hls[:, :, 2])
    skew_s = skew(hls[:, :, 2], axis=None)
    kurt_s = kurtosis(hls[:, :, 2], axis=None)
    
    contrast = np.std((hls[:, :, 1] - min_l) / (max_l - min_l))
    
    return [min_l, max_l, std_l, avg_l, skew_l, kurt_l,
            min_s, max_s, std_s, avg_s, skew_s, kurt_s, contrast]


def colorfulness2(img):
    mean_img = np.mean(img, axis=2)

    return np.mean(np.sqrt((img[:, :, 0] - mean_img) ** 2
                           + (img[:, :, 1] - mean_img) ** 2
                           + (img[:, :, 2] - mean_img) ** 2))


def colorfulness(img):
    rg = img[:, :, 0] - img[:, :, 1]
    yb = (img[:, :, 0] + img[:, :, 1]) / 2 - img[:, :, 2]

    return np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)


def dullness_lightness(img):
    dark_px = 0
    light_px = 0
    max_img = np.max(img, axis=2)
    min_img = np.min(img, axis=2)
    dark_px = np.sum(max_img < 20)
    light_px = np.sum(min_img > 240)

    return (dark_px / (img.shape[0] * img.shape[1]), light_px / (img.shape[0] * img.shape[1]))


def invariant_moments(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.HuMoments(cv2.moments(img2))


def ed(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def distances_between_centroids(img):
    r_m = cv2.moments(img[:, :, 0])
    g_m = cv2.moments(img[:, :, 1])
    b_m = cv2.moments(img[:, :, 2])
    
    r_x = r_m['m10'] / r_m['m00']
    r_y = r_m['m01'] / r_m['m00']
    
    g_x = g_m['m10'] / g_m['m00']
    g_y = g_m['m01'] / g_m['m00']
    
    b_x = b_m['m10'] / b_m['m00']
    b_y = b_m['m01'] / b_m['m00']
    
    d1 = ed(r_x, r_y, g_x, g_y)
    d2 = ed(r_x, r_y, b_x, b_y)
    d3 = ed(b_x, b_y, g_x, g_y)
    
    return (d1, d2, d3, (d1 + d2 + d3)/3)


def features_1(fname):
    img = cv2.imread(fname)
    if img is not None:
        output = [getSize(fname), getDimensions(img), get_blurrness_score(img), colorfulness(img),
                  colorfulness2(img),  # 0-4 (scalar, tuple of 2 elements, scalars)
                  get_hsv_features(img),  # 5 - list of dimension 13
                  get_rgb_features(img),   # 6 - list of dimension 18
                  dullness_lightness(img),  # 7 - tuple of dimension 2
                  invariant_moments(img),  # 8 - HU moments, dimension 7
                  distances_between_centroids(img)]  # 9 - distances between R/G/B centroids, dimension 4
    else:
        bad_img_names.append(fname)
        return [0, [0, 0], 0, 0, 0, [0] * 13, [0] * 18, [0, 0], [0] * 7, [0] * 4]

    return output


bad_img_names = []
tqdm.pandas(desc="Progress")


n_imgs = 0

pth = 'Avito/images/data/competition_files/test_jpg/'
imgs = os.listdir(pth)
print(len(imgs))

imgs = [x for x in imgs if x.endswith('.jpg')]
basic_test = {'images': imgs}
df = pd.DataFrame(basic_test)
df['filename'] = df['images'].progress_apply(lambda x: pth+x)

start = perf_counter()
df['features'] = df['filename'].progress_apply(lambda x: features_1(x))
print(perf_counter() - start)

print(len(df))
print(len(bad_img_names))

for bad_img in bad_img_names:
    df = df[df['filename'] != bad_img]
print(len(df))

df.drop(('filename'), axis=1, inplace=True)
df['images'] = df['images'].apply(lambda x: x[:-4])

df['size'] = df['features'].apply(lambda x: x[0])
df['area'] = df['features'].apply(lambda x: x[1][0])
df['w_h_ratio'] = df['features'].apply(lambda x: x[1][1])
df['blurness'] = df['features'].apply(lambda x: x[2])
df['colorfulness'] = df['features'].apply(lambda x: x[3])
df['colorfulness2'] = df['features'].apply(lambda x: x[4])

i = 0
for channel_name in ['l', 's']:
    for function_name in ['min', 'max', 'std', 'avg', 'skew', 'kurt']:
        df[channel_name + '_' + function_name] = df['features'].apply(lambda x: x[5][i])
        i += 1
df['contrast'] = df['features'].apply(lambda x: x[5][12])

i = 0
for channel_name in ['r', 'g', 'b']:
    for function_name in ['min', 'max', 'std', 'avg', 'skew', 'kurt']:
        df[channel_name + '_' + function_name] = df['features'].apply(lambda x: x[6][i])
        i += 1
        
df['dullness'] = df['features'].apply(lambda x: x[7][0])
df['lightness'] = df['features'].apply(lambda x: x[7][1])

for i in range(7):
    df['hu_mom_' + str(i)] = df['features'].apply(lambda x: x[8][i][0])
    
df['dist_centroid_rg'] = df['features'].apply(lambda x: x[9][0])
df['dist_centroid_rb'] = df['features'].apply(lambda x: x[9][1])
df['dist_centroid_bg'] = df['features'].apply(lambda x: x[9][2])
df['dist_centroid_avg'] = df['features'].apply(lambda x: x[9][3])

df.drop(('features'), axis=1, inplace=True)
df.to_csv('img_features_test.csv', index=False)

