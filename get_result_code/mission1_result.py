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

# ===========학습된 신경망을 불러오기 위한 클래스 부분===========#

# Resnet 블럭과 Resnet 객체 모두에서 사용하는 3x3 컨볼루션 함수
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block 객체
# 이를 층층이 쌓아 올려서 Residual Network를 구성하게 된다.
class ResidualBlock(nn.Module):
    # 3x3 컨볼루선, 배치 노멀라이제이션, 렐루를 2번 반복하여 쌓아준다.
    # 그 다음 결과 벡터의 크기를 downsample 설정 값 만큼줄여준다.
    # 줄여주는 이유는 신경망이 지나치게 깊어질 경우
    # 연산량이 너무 많아서 학습이 제대로 이루어지지 않을 경우를 방지하기 위함이다.
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    # 신경망을 앞으로 통과하는 함수
    # 앞서 쌓은 연산들을 순차적으로 수행하게 된다.
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 이전 네트워크에서 전달받은 잔여 값을 결과 값에 더해준다.
        if self.downsample:
            residual = self.downsample(x)
        out += residual

        # 이 더한 값에 다시 활성화 함수를 실행시켜 최종 결과 값을 반환한다.
        out = self.relu(out)
        return out

# Residual block을 쌓아올려 Residual Network를 구성하는 부분
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, input=None):
        super(ResNet, self).__init__()

        # 맨 처음 입력 이미지에 컨볼루션, 배치 노멀라이제이션,렐루를 실행하는 부분
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 본격적으로 Residual block을 쌓아 올리는 부분
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)

        # 입력 이미지의 크기에 따라서 가장 마지막 단의 Fully Connected Layer의 크기를 조절해준다.
        self.fc_size = 256
        if input==64 :
            self.fc_size=256
        elif input==128 :
            self.fc_size=1024
        elif input==256 :
            self.fc_size=4096
        self.fc = nn.Linear(self.fc_size, num_classes)

    # Residual Block 객체를 만드는 부분
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

    # Residual Network를 앞으로 통과하여 합성일 확률을 추출하는 부분
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

# 코드 실행 시 합성인지 여부를 판별할 입력 이미지 디렉터리를 전달받는 부분
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    dest="input_dir",
                    default="input_dir",
                    help="Input directory",
                    required=True)

args = parser.parse_args()
input_dir = args.input_dir

# 64x64 크기 이미지들의 합성 여부를 판별하는 모델 로딩
model_64_array=[]
for i in range(1, 4) :
    resnet_64 = ResNet(ResidualBlock, [3, 3, 3], input=64)
    resnet_64.cuda()
    model_path = '../mission1_models/mission1_64_'+str(i)+'.pth'
    resnet_64.load_state_dict(torch.load(model_path))
    model_64_array.append(resnet_64)

# 128x128 크기 이미지들의 합성 여부를 판별하는 모델 로딩
model_128_array=[]
for i in range(1, 4) :
    resnet_128 = ResNet(ResidualBlock, [3, 3, 3], input=128)
    resnet_128.cuda()
    model_path='../mission1_models/mission1_128_'+str(i)+'.pth'
    resnet_128.load_state_dict(torch.load(model_path))
    model_128_array.append(resnet_128)

# 256x256 크기 이미지들의 합성 여부를 판별하는 모델 로딩
model_256_array=[]
for i in range(1,4) :
    resnet_256 = ResNet(ResidualBlock, [3, 3, 3], input=256)
    resnet_256.cuda()
    model_path='../mission1_final_models/mission1_256_'+str(i)+'.pth'
    resnet_256.load_state_dict(torch.load(model_path))
    model_256_array.append(resnet_256)

# 결과 값 저장할 디렉터리 경로, 만일 없을 시 생성
output_dir = '../result/'
if not os.path.exists(output_dir) :
    os.mkdir(output_dir)

# 결과 저장할 파일 생성
mission1_result=open(output_dir+'mission1_result.txt', 'w')

file_array=os.listdir(input_dir)
file_array.sort()

# 이미지 크기를 조절하기 위한 크기 설정
max_64=(64,64)
max_128=(128,128)
max_256=(256,256)

for file in file_array :
    # 입력 디렉터리로부터 이미지들을 읽어옴
    cur_img = input_dir+file
    img = cv2.imread(cur_img)
    rows, cols, channels = img.shape

    # 입력 이미지의 크기에 맞게 64x64, 128x128, 256x256 크기로 조절해준다.
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

    # 결과 값을 파일에 기록해준다.
    mission1_result.write(filename+','+str(prob)+'\n')
    print(filename+','+str(prob))




