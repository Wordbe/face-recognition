import os
import argparse
import dataloader
import torch
import torch.nn as nn
import torchvision
# from efficientnet_pytorch import EfficientNet
# import pretrainedmodels

#####################################
# Settings for saving model, board  #
#####################################
from datetime import date, datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


#####################################
# Hyper parmeters                   #
#####################################
N_CLASS = 6
LEARNING_RATE = 1e-4
N_EPOCH = 300
BATCH_SIZE = 64


def train(dataloader, root, device):
    #####################################
    # Set Model                         #
    #####################################
    # Using torch.hub
    # 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', resnext101_32x8d', 'shufflenet_v2_x1_0'
    model_name = 'resnet18'
    model = torch.hub.load('pytorch/vision:v0.5.0', model_name, pretrained=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=N_CLASS, bias=True)
    
    # Using efficientnet_pytorch
    # model_name = 'efficientnet-b7'
    # model = EfficientNet.from_pretrained(model_name, num_classes=N_CLASS)
    
    # Using pretrainedmodels
    # 'se_resnext101_32x4d', 'senet154'
    # model_name = 'senet154'
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # last_linear_infeature = model.last_linear.in_features
    # model.last_linear = nn.Linear(in_features=last_linear_infeature, out_features=N_CLASS)
    # model.fc = nn.Linear(in_features=model.fc.in_features, out_features=N_CLASS, bias=True)
    
    model = model.to(device)
    
    #####################################
    # Set Loss and Optimizer            #
    #####################################
    num_labels_ratio = [0.044, 0.658, 0.240, 1.000, 1.000, 0.152]
    ce_weight = torch.tensor(num_labels_ratio).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    
    # Model save name
    today = date.today().strftime("%y%m%d")
    now = datetime.now().strftime("%y%m%d_%H%M")
    weight_name = now + \
                  '_' + model_name + \
                  '_batch' + str(BATCH_SIZE) + \
                  '_lr' + str(LEARNING_RATE).split('.')[-1] + \
                  '_epoch' + str(N_EPOCH)
    if not os.path.isdir(root + '/weights/' + today):
      os.makedirs(root + '/weights/' + today)
    weight_savename = root + '/weights/' + today + '/' + weight_name + '.pth'

    # Tensorboard for logging
    if not os.path.isdir(root + '/tensorboard/' + today):
      os.makedirs(root + '/tensorboard/' + today)
    writer = SummaryWriter(root + '/tensorboard/' + today + '/' + weight_name)


    #####################################
    # Train                             #
    #####################################
    train_loader, valid_loader, num_trainset, num_validset = dataloader
    
    min_loss = float('inf')
    valid_min_loss = float('inf')
    valid_max_acc = float('-inf')
    
    for epoch in range(N_EPOCH):

      # Switch model to train mode
      model.train(); 
      loss_per_epoch, correct = 0, 0
      for batch in tqdm(train_loader):
        imgs = batch['img'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)

        # clear gradient accumulators
        optimizer.zero_grad()

        outs = model(imgs)
        _, preds = torch.max(outs, 1)
        loss = criterion(outs, labels)

        # Backpropagate and update optimizer
        loss.backward(); optimizer.step()

        loss_per_epoch += loss.item()
        correct += (preds == labels).float().sum().item()

        # # Save model when loss decreases
        # if loss.item() < min_loss:
        #     torch.save(model, weight_savename)
        #     min_loss = loss.item()

      loss_per_epoch /= len(train_loader)
      accuracy = 100 * correct / num_trainset
      print('\t epoch: {}, train loss: {}. train accuracy: {}'.format(epoch + 1, loss_per_epoch, accuracy))

      # Evalute model on validation set every epoch
      # Switch model to evaluate mode
      model.eval()
      valid_loss_per_epoch, valid_correct = 0, 0
      with torch.no_grad():
        for valid_batch in tqdm(valid_loader):
          valid_imgs = valid_batch['img'].to(device, dtype=torch.float)
          valid_labels = valid_batch['label'].to(device)

          valid_outs = model(valid_imgs)
          _, valid_preds = torch.max(valid_outs, 1)
          valid_loss = criterion(valid_outs, valid_labels)
          valid_loss_per_epoch += valid_loss.item()
          valid_correct += (valid_preds == valid_labels).float().sum()

      valid_loss_per_epoch /= len(valid_loader)
      valid_accuracy = 100 * valid_correct / num_validset

      print('\t epoch: {}, valid loss: {}. valid accuracy: {}'.format(epoch + 1, valid_loss_per_epoch, valid_accuracy))

      if valid_accuracy > valid_max_acc:
          torch.save(model, weight_savename)
          print("validation accuracy {} --> {}, model saved".format(valid_max_acc, valid_accuracy))
          valid_max_acc = valid_accuracy

      # Log on the tensorboard
      writer.add_scalars('loss/train+validation',
                          {'train_loss' : loss_per_epoch, 'valid_loss' : valid_loss_per_epoch},
                          epoch + 1 )
      writer.add_scalars('accuracy/train+validation',
                          {'train_acc' : accuracy, 'valid_acc' : valid_accuracy},
                          epoch + 1 )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="A root path of the train data is needed")
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    label_csv = args.root + '/face_classification_csv/ml_8_faceclassifier_train.csv'
    data_loader = dataloader.load_data(args.root, label_csv, BATCH_SIZE)
    
    train(data_loader, args.root, device)
    print("Train finished successfully.")
    