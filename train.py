import torch
from PIL import Image
import os
import glob
from torch import nn
from torch import optim
from torch.utils.data import Dataset, dataset
import random
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.tensorboard.summary import image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
from torchvision import models

myWriter = SummaryWriter('D:\pytorch\pytorch基础\logs')


class Mydataset(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('    '), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224.0 / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new('RGB', (size_bg, size_bg))
        img_bg.paste(img_bg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)


train_dataset = Mydataset("D:\pytorch\Dataset/train.txt", True)
test_dataset = Mydataset("D:\pytorch\Dataset/val.txt", False)
# print('个数：', len(train_dataset))

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=16, shuffle=True)

# for image, label in test_loader:
#     print(image.shape)
#     print(label)

# 定义模型
myModel = torchvision.models.resnet50(pretrained=True)
# 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 2)

# 损失函数及优化器
# GPU加速
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

learning_rate = 0.001
myOptimzier = optim.Adam(myModel.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    myOptimzier, step_size=30, gamma=0.1)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(10):
    training_loss = 0.0
    scheduler.step()
    for _step, input_data in enumerate(train_loader):

        image, label = input_data[0].to(
            myDevice), input_data[1].to(myDevice)  # GPU加速
        predict_label = myModel.forward(image)

        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss,
                            global_step=_epoch * len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()

        myOptimzier.step()

        training_loss = training_loss + loss.item()
        if _step % 10 == 0:
            print('[iteration - %3d] training loss: %.3f' %
                  (_epoch * len(train_loader) + _step, training_loss / 10))
            training_loss = 0.0
            print()
    correct = 0
    total = 0
    torch.save(myModel, 'Resnet50_Own.pkl')  # 保存整个模型
    myModel.eval()
    for images, labels in test_loader:
        # GPU加速
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Testing Accuracy : %.3f %%' % (100 * correct / total))
    myWriter.add_scalar('test_Accuracy', 100 * correct / total)
