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
from src import detect_faces, show_bboxes
from PIL import Image, ImageOps
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

        if input==128 :
            self.fc_size=1024

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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    h, w = image.shape[:2]
    dim = (width, height)
    if h > w :
        r = height / float(h)
        dim = (int(w * r), height)

    else :
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    h, w = resized.shape[:2]

    left = int((width - w)/2)
    right = int((width - w)/2)
    top = int((height - h)/2)
    bottom = int((height - h)/2)
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,(0,0,0))

    if image.shape[:2]!=(128,128) :
        image = cv2.resize(image, (128,128))

    return image

def model_forward(cur_model, input_dir, id_list) :

    result_array=[]
    for id in id_list:

        # 이미지 안에 포함된 각 얼굴별 최종 확률을 담은 배열
        total_prob = []

        # 얼굴 이미지 하나씩 현재 모델을 통과해서 확률을 계산
        for i in range(4):
            cur_img_path = input_dir + id + '_' + str(i) + '.jpg'

            if not os.path.exists(cur_img_path):
                break

            # 얼굴 이미지를 읽어온다
            img = cv2.imread(cur_img_path)

            # 이미지를 파이토치가 인식할 수 있는 형태로 변환해준다
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).float()

            # GPU 연산을 할 수 있도록 설정한다
            img = Variable(img.cuda())

            # 현재 모델을 통과해서 합성일 확률을 가져온다
            vector = cur_model.forward(img)

            # 가져온 값을 보기 편한 형태로 변환하여 준다
            vector = vector.cpu()
            vector = F.softmax(vector, dim=1)
            vector = vector.data.numpy().squeeze()
            vector = np.expand_dims(vector, axis=0)

            total_prob.append(vector[0][1])

        # 만일 이미지 안에서 검출된 얼굴이 없다면 합성일 확률 1.0을 넣어준다
        if len(total_prob) == 0:
            prob = str(1.0000)

        # 얼굴이 포함되어 있다면 계산한 확률을 결과 값에 추가한다
        else:
            prob = format(max(total_prob),'.4f')

        result_array.append(prob)

    return result_array

# 미리 잘라놓은 얼굴 이미지가 저장된 디렉토리
input_dir = '../mission2_faces/'

# resnet 객체를 생성한 뒤에, 해당 모델을 로딩한다
model_dict = {}
model_name_array = ['MOUTH_blur_1.pth', 'EYES_blur.pth','LEFT_EYE_blur.pth','LEFT_EYE_cut.pth','RIGHT_EYE_blur.pth', 'RIGHT_EYE_cut.pth', 'MOUTH_blur_2.pth']
for model in model_name_array :
    resnet_128 = ResNet(ResidualBlock, [3, 3, 3], input=128)
    resnet_128.cuda()
    model_path = '../mission2_models/'+model
    resnet_128.load_state_dict(torch.load(model_path))
    model_dict.update({model:resnet_128})

# 미리 잘라놓은 얼굴 이미지를 읽어오기 위한 이미지 id 리스트
id_list = []
id_file = open('id_list.txt', 'r')
for row in id_file:
    row = row.replace('\n','')
    id_list.append(row.split(',')[0])

# 각 모델별로 산출한 결과 값과 모델 이름을 연결해주는 딕셔너리
result_dict={}

# 이제 로딩한 모델들에 이미지들을 순차적으로 통과시키면서 결과 값 배열을 가져온다
for model_name in model_name_array :
    print(model_name, ' 통과 중')
    cur_model = model_dict.get(model_name)
    result_array = model_forward(cur_model, input_dir, id_list)
    result_dict.update({model_name:result_array})

# 최종 결과 값은 입 번짐 필터 결과 값으로 기본 설정
final_result = result_dict.get('MOUTH_blur_2.pth')

# 얼굴 절반 번짐 검출 필터에서 0.999 이상인 데이터 필터링
for i in range(len(final_result)) :
    cur_result = result_dict.get('MOUTH_blur_1.pth')
    if float(cur_result[i]) >= 0.99 :
        final_result[i] = cur_result[i]

# 왼쪽 눈 블러 필터에서 0.999 이상 합성일 확률 이미지 필터링
for i in range(len(final_result)) :
    cur_result = result_dict.get('LEFT_EYE_blur.pth')
    if float(cur_result[i]) >= 0.99 :
        final_result[i] = cur_result[i]

# 왼쪽 눈 컷 필터에서 0.99 이상 결과 값 추가
for i in range(len(final_result)) :
    cur_result = result_dict.get('LEFT_EYE_cut.pth')
    if float(cur_result[i]) >= 0.99 :
        final_result[i] = cur_result[i]

# 양쪽 눈 블러 필터에서 0.999 이상 합성일 확률 이미지 필터링
for i in range(len(final_result)) :
    cur_result = result_dict.get('EYES_blur.pth')
    if float(cur_result[i]) >= 0.999 :
        final_result[i] = cur_result[i]

# 오른쪽 눈 블러 필터에서 0.999 이상 결과 값 추가
for i in range(len(final_result)) :
    cur_result = result_dict.get('RIGHT_EYE_blur.pth')
    if float(cur_result[i]) >= 0.999 :
        final_result[i] = cur_result[i]

# 오른쪽 눈 컷 필터에서 0.999 이상 결과 값 추가
for i in range(len(final_result)) :
    cur_result = result_dict.get('RIGHT_EYE_cut.pth')
    if float(cur_result[i]) >= 0.999 :
        final_result[i] = cur_result[i]

# 결과값 저장할 파일 생성
mission2_result = open('../result/mission2_result.txt', 'w')

for i, id in enumerate(id_list) :
    mission2_result.writelines(id + ',' + final_result[i] + '\n')
