# Train set

import pandas as pd
root = 'D:/dataset/face-classification/'
train_csv = pd.read_csv(root + 'face_classification_csv/ml_8_faceclassifier_train.csv')
test_csv = pd.read_csv(root + 'face_classification_csv/ml_8_faceclassifier_test.csv')
train_dir = root + 'face_images_128x128/'
train_paths = list(train_dir + train_csv['filename'])
test_paths = list(train_dir + test_csv['filename'])

import numpy as np
import matplotlib.pyplot as plt
# Train labels
train_labels = list(train_csv['label'])
hists = np.unique(train_labels, return_counts=True)
plt.hist(train_labels, rwidth=0.8)
plt.show()

print(hists)
'''
(array([0, 1, 2, 4, 5]), array([ 116, 1727,  631, 2626,  400], dtype=int64))
'''

print(hists[1] / 2626)
'''
array([0.04417365, 0.65765423, 0.24028941, 1.        , 0.15232292])
'''

# Validation set
import glob
root = 'D:/dataset/face-classification'
train_dir = root + '/face_images_128x128/'
train_csv = glob.glob(root + '/face_classification_csv/*train.csv')[0]

# Split train and validation data
train_table = pd.read_csv(train_csv)
valid_table = train_table.sample(frac=0.2, random_state=999)
# train_table.drop(index=valid_table.index, axis=0, inplace=True)

# Reset index (0, 1, 2, ...)
# train_table.reset_index(inplace=True)
valid_table.reset_index(inplace=True)

valid_labels = list(valid_table['label'])
valid_hists = np.unique(valid_labels, return_counts=True)
print(valid_hists)
plt.hist(valid_labels, rwidth=0.8)
plt.show()
print(valid_hists[1] / 522)
'''
(array([0, 1, 2, 4, 5]), array([ 26, 355, 123, 522,  74], dtype=int64))
[0.04980843 0.68007663 0.23563218 1.         0.14176245]
'''