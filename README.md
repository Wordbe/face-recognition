# face-recognition
**Face classification** :cowboy_hat_face:



---



### **Face Images**

Train set : 5500

Test set : 1500

128 x 128 x 3







### Models

| Model             | Accuracy (%) |
| ----------------- | ------------ |
| InceptionResNetV2 | 94.5         |
| ResNext101_32x8d  | 94.1         |
| SENet154          | 94           |
| EfficientNet-b7   | 93.1         |
| ResNet152         | 92.9         |





### Method

* Data normalization

  RGB mean, std regularization

* Data augmentation

  Horizontal Flip, Brightness, Contrast, Saturation, Hue

* Imbalanced Labels

  Applied weighted loss

* Ensemble

  Above 5 models



### Requirements

```
torch 1.3.0
torchvision 0.4.1
Pillow 6.2.0
```



### Get started



**Train**

```
$ python train.py --root dataset/face-classification
```

models and logs are saved in weight and tensorboard directory



**Test**

```
$ python test.py --root dataset/face-classification
```

output file (submission.csv) are saved in root directory



### Weights

The models are saved as a pth file in  `root/weights`.



### Tensorboard

The loggings are saved in `root/tensorboard`

