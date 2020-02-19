#####################################
# Test data                         #
#####################################
import os
import glob
from datetime import date, datetime
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from scipy import stats
import argparse

class FaceDatasetTest(Dataset):
  def __init__(self, test_table, test_dir):
    self.test_dir = test_dir
    self.test_table = test_table
      
  def __len__(self):
    return len(self.test_table)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist() # to list
    
    filename = self.test_table['filename'][idx]
    img_file = self.test_dir + filename
    img = Image.open(img_file).convert('RGB')

#     img299 = transforms.Compose([transforms.Resize((299, 299)),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize(mean=[0.5325000360890079, 0.4324560664275582, 0.38310616926626534],
#                                                       std=[0.28341811024667896, 0.2566113920988699, 0.2580127451979737])])(img)
    
#     img224 = transforms.Compose([transforms.Resize((224, 224)),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize(mean=[0.5325000360890079, 0.4324560664275582, 0.38310616926626534],
#                                                       std=[0.28341811024667896, 0.2566113920988699, 0.2580127451979737])])(img)

    img = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5325000360890079, 0.4324560664275582, 0.38310616926626534],
                                                   std=[0.28341811024667896, 0.2566113920988699, 0.2580127451979737])])(img)
#     sample = {'img': [img, img299, img224, img, img], 'filename': filename}
    sample = {'img': [img], 'filename': filename}

    return sample

def load_test_data(root, label_csv):
    test_dir = root + '/face_images_128x128/'
    test_csv = glob.glob(label_csv)[0]

    test_table = pd.read_csv(test_csv)

    # Reset index (0, 1, 2, ...)
    test_table.reset_index(inplace=True)

    test_dataset = FaceDatasetTest(test_table=test_table, test_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("test_loader: {}".format(len(test_loader)))

#     for i, t in enumerate(test_loader):
#       print(i, len(t['img']), t['filename'])
#       imgs = t['img']
#       for img in imgs:
#         test_npgrid = torchvision.utils.make_grid(img).numpy()
#         print(np.shape(test_npgrid))
#         plt.figure(figsize=(3, 4))
#         plt.imshow(np.transpose(test_npgrid, (1, 2, 0)), interpolation='nearest')
#         plt.show()
#       break
    return test_loader

if __name__ == '__main__':
    #####################################
    # Test                              #
    #####################################
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="A root path of the train data is needed")
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    root = args.root
    label_csv = root + '/face_classification_csv/ml_8_faceclassifier_test.csv'

    model_names = glob.glob(root + '/weights/**/*.pth')
    test_models = [torch.load(m) for m in model_names]

    test_loader = load_test_data(args.root, label_csv)
    
    filenames = []
    predictions = []
    cnt = 0
    for t in tqdm(test_loader):
      test_filename = t['filename'][0]

      #####################################
      # Ensemble(Vote)                    #
      #####################################
      preds = []
      for img, model in zip(t['img'], test_models):
        test_img = img.to(device, dtype=torch.float)
        model.eval()
        with torch.no_grad():
          test_out = model(test_img)
          _, test_pred = torch.max(test_out, 1)
        preds.append(test_pred.item())
      final_pred = stats.mode(np.array(preds))[0][0]

      filenames.append(test_filename)
      predictions.append(final_pred)

    today = date.today().strftime("%y%m%d")
    submission = root + '/' + today + '_submission.csv'
    pd.DataFrame({'filename': filenames, 'prediction': predictions}).set_index('filename').to_csv(submission)