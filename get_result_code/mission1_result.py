import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import pandas as pd
import cv2
import numpy as np
import random
import sys, os
import torch.nn.functional as F
import argparse

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, input=None):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc_size=0

        if input==64 :
            self.fc_size=256
        elif input==128 :
            self.fc_size=1024
        elif input==256 :
            self.fc_size=4096

        self.fc = nn.Linear(self.fc_size, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# get input dir which passed from cmd lines
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    dest="input_dir",
                    default="input_dir",
                    help="Input directory")

args = parser.parse_args()
input_dir = args.input_dir

# load model ensamble for input size 64
model_64_array=[]
for i in range(1, 4) :
    resnet_64 = ResNet(ResidualBlock, [3, 3, 3], input=64)
    resnet_64.cuda()
    model_path = '../mission1_models/mission1_64_'+str(i)+'.pth'
    resnet_64.load_state_dict(torch.load(model_path))
    model_64_array.append(resnet_64)

# load model ensamble for input size 128
model_128_array=[]
for i in range(1, 4) :
    resnet_128 = ResNet(ResidualBlock, [3, 3, 3], input=128)
    resnet_128.cuda()
    model_path='../mission1_models/mission1_128_'+str(i)+'.pth'
    resnet_128.load_state_dict(torch.load(model_path))
    model_128_array.append(resnet_128)

# load model ensamble for input size 256
model_256_array=[]
for i in range(1,4) :
    resnet_256 = ResNet(ResidualBlock, [3, 3, 3], input=256)
    resnet_256.cuda()
    model_path='../mission1_models/mission1_256_'+str(i)+'.pth'
    resnet_256.load_state_dict(torch.load(model_path))
    model_256_array.append(resnet_256)

# create txt file that save the predict results
mission1_result=open('../result/mission1_result.txt', 'w')

file_array=os.listdir(input_dir)
file_array.sort()

# maximum sizes of imgs
max_64=(64,64)
max_128=(128,128)
max_256=(256,256)

for file in file_array :
    # read img from input dir
    cur_img = input_dir+file
    img = cv2.imread(cur_img)
    rows, cols, channels = img.shape

    # resize the input img into certain maximum size
    if rows<128 and cols<128 :
        img = cv2.resize(img, max_64)
        cur_model_array = model_64_array

    elif rows<256 and cols<256 :
        img = cv2.resize(img, max_128)
        cur_model_array = model_128_array

    else :
        img = cv2.resize(img, max_256)
        cur_model_array = model_256_array

    # 파이토치가 인식할 수 있는 형태로 바꾸어주기
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()

    # GPU 사용이 가능하도록 변형
    img = Variable(img.cuda())

    prob_array=[]
    for model in cur_model_array :

        # 입력 이미지의 크기에 맞는 모델 앙상블을 통과하면서 합성일 확률 산출
        vector = model.forward(img)

        # 산출된 벡터를 CPU 계산이 가능하게끔 형 변환해준 뒤, 소프트 맥스 함수 적용
        vector = vector.cpu()
        vector = F.softmax(vector, dim=1)

        # 이를 주최측이 요구하는 소수점 4째 자리까지 소수로 표현
        vector = vector.data.numpy().squeeze()
        vector = np.expand_dims(vector, axis=0)
        prob=vector[0][1]

        prob_array.append(prob)

    prob_sum = sum(prob_array)
    prob_count = len(prob_array)
    if prob_count== 0 :
        prob = 1.0
    else :
        # 모델 앙상블이 반환한 확률 값들의 평균 치를 구한다.
        prob = format(prob_sum/prob_count, '.4f')

    filename=file.split('.')[0]

    #write the result into txt file
    mission1_result.write(filename+','+str(prob)+'\n')
    print(filename+','+str(prob))




