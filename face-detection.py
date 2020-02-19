#####################################
# Test data                         #
#####################################
class FaceDatasetTest(Dataset):
  def __init__(self, test_table, test_dir):
    self.test_dir = test_dir
    self.test_table = test_table
    
    self.H = 128 # Height
    self.W = 128 # Width
      
  def __len__(self):
    return len(self.test_table)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist() # to list
    
    filename = self.test_table['filename'][idx]
    img_file = self.test_dir + filename
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # numpy array to tensor
    img = torch.tensor(img.transpose((2, 0, 1)))
    sample = {'img': img, 'filename': filename}

    return sample

root = 'drive/My Drive/team10'
test_dir = root + '/face_images_128x128/'
test_csv = glob.glob(root + '/csvs/*test.csv')[0]

test_table = pd.read_csv(test_csv)

# Reset index (0, 1, 2, ...)
test_table.reset_index(inplace=True)

test_dataset = FaceDatasetTest(test_table=test_table, test_dir=test_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(len(test_loader))

for i, t in enumerate(test_loader):
  print(i, np.shape(t['img']), t['filename'])
  test_npgrid = torchvision.utils.make_grid(t['img']).numpy()
  plt.figure(figsize=(3, 4))
  plt.imshow(np.transpose(test_npgrid, (1, 2, 0)), interpolation='nearest')
  plt.show()
  break

#####################################
# Object detector                   #
#####################################

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
num_classes = 91 # coco dataset classes + background
detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
detection_model.to(device)
detection_model.eval()
    
#####################################
# Detection Test                    #
#####################################

from tqdm import tqdm
import torch

detection_filenames = []
detection_predictions = []

cnt = 0
for i in tqdm(test_loader):
  detection_test_img = i['img']
  detection_test_npgrid = torchvision.utils.make_grid(detection_test_img).numpy()
  plt.figure(figsize=(3, 4))
  plt.imshow(np.transpose(detection_test_npgrid, (1, 2, 0)), interpolation='nearest')
  plt.show()

  detection_test_img = detection_test_img.to(device, dtype=torch.float)
  detection_test_filename = i['filename']

  detection_test_out = detection_model(detection_test_img)[0]
  print("개수: ", len(detection_test_out['labels']))
  print("boxses : ", detection_test_out['boxes'])
  print("labels : ", detection_test_out['labels'])
  print("scores : ", detection_test_out['scores'])
  
  if cnt == 10:
    break
  cnt += 1