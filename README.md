# Image-classification---emotion
## Description
#### Crop된 얼굴의 감정을 판단하는 네트워크를 개발하는 Task

## DB 및 backbone
#### Image Detection
#### https://www.kaggle.com/ananthu017/emotion-detection-fer
#### backbone code - ResNet
####  https://github.com/Jongchan/week1_tiny_imagenet_tutorial

## Result
#### 최고 성능
1. ResNet18
2. batch_size = 128
3. learning_rate = 0.01
4. learning_drop_rate = 0.05
5. epoch = 50
6. Augmentation : random_rotation → 50, random_perspective
7. normalize (0.6)
8. loss = 0.71, val_acc1 = 67.54

