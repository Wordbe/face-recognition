import pandas as pd
root = 'D:/dataset/face-classification/'
train_csv = pd.read_csv(root + 'face_classification_csv/ml_8_faceclassifier_train.csv')
test_csv = pd.read_csv(root + 'face_classification_csv/ml_8_faceclassifier_test.csv')
train_dir = root + 'face_images_128x128/'

from IPython.display import clear_output
import numpy as np
import cv2

def set_range_zero2one(img):
    result_img = img - np.min(img)
    result_img = result_img / np.max(result_img)
    return result_img

def get_mean_std_from_allImg(train_paths):
    '''
    모든 이미지셋 픽셀 값의 평균과 표준편차를 구함.
    '''
    colors = ['b', 'g', 'r']
    m = {c: [] for c in colors}
    v = {c: [] for c in colors}
    
    for i, path in enumerate(train_paths):
        img = cv2.imread(path)
        img = set_range_zero2one(img)
        n_color = img.shape[2]
        
        for c, nth in zip(colors, range(n_color)):
            m[c].append(np.mean(img[:, :, nth]))
            v[c].append(np.var(img[:, :, nth]))
        print('{}th done'.format(i+1))
        clear_output(wait=True)
    
    result_mean, result_std = [], []
    
    for c in colors:
        M = np.array(m[c])
        V = np.array(v[c])
        tot_m = np.mean(M)
        tot_s = np.sqrt((np.sum(M*M) + np.sum(V)) / len(M) - tot_m*tot_m)
        result_mean.append(tot_m)
        result_std.append(tot_s)
    return {'mean': result_mean, 'std': result_std}


train_paths = list(train_dir + train_csv['filename'])
test_paths = list(train_dir + test_csv['filename'])
train_result = get_mean_std_from_allImg(train_paths)
test_result = get_mean_std_from_allImg(test_paths)

'''
train_result, test_result
({'mean': [0.37977362891997446, 0.42738061407253225, 0.5239860024529774],
  'std': [0.25590995105180836, 0.25519093913452706, 0.2834680229098988]},
 {'mean': [0.38310616926626534, 0.4324560664275582, 0.5325000360890079],
  'std': [0.2580127451979737, 0.2566113920988699, 0.28341811024667896]})
'''