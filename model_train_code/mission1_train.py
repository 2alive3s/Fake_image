# 생성자 : 김형준
# 최종 수정일 : 18. 6. 29.
# 파일 개요 : 임무 1 합성 이미지 필터 신경망 모델과 학습 코드를 담고 있는 파일

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import pandas as pd
import cv2
import numpy as np
import random
import sys
import argparse

# ===========학습에 필요한 데이터를 읽어오는 데이터 셋 클래스===========#

# 신경망을 학습시킬 때, 미리 만들어 놓은 이미지 데이터 셋을 불러오는 데이터 셋 클래스
# 0과 1로 라벨링된 이미지 데이터를 신경망이 학습 가능한 텐서 형식으로 변환하여 반환한다.
# isTrain 파라미터에 따라 학습용 이미지를 불러올 것인지, 테스트 용 이미지를 불러올 것인지가 결정된다
class CustomDataset(Dataset) :
    def __init__(self, isTrain=None, train_label=None, test_label=None, input_size=None):
        # 입력 이미지로 받을 크기 설정
        self.input_size = input_size

        # 학습용 데이터를 불러올 것인지, 테스트용 데이터를 불러올 것인지에 따라 각각 csv 파일을 읽어온다.
        # train_label과 test_label은 각각 학습과 테스트에 사용될 이미지들의
        # 경로 값과 진짜인지 가짜인지를 나타내는 라벨 정보를 포함한다
        if isTrain :
            self.image_label = pd.read_csv(train_label, delimiter=',')
            print(train_label)

        else :
            self.image_label = pd.read_csv(test_label , delimiter=',')
            print(test_label)


    # 데이터 로더가 데이터를 읽어올 때 호출되는 함수
    # 특정 인덱스의 이미지 데이터, 라벨은 무엇인지를 반환한다.
    def __getitem__(self, index):

        # 먼저 앞서 설정한 image_label에서 읽어오고자 하는 이미지 경로를 가져온다.
        # 그 다음 openCV를 활용하여 이미지를 숫자 배열의 형태로 변환한다.
        img_path = self.image_label.iloc[index, 1]
        img = cv2.imread(img_path)

        # 다음으로 trainsform 함수를 호출하여 이미지의 크기, 각도를 변환한다.
        # 이렇게 변환하는 이유는 신경망이 다양한 형태의 이미지를 학습하도록 하여 정확도를 높이기 위함이다.
        img = self.transform_img(img, input_size)

        # 이미지의 RGB 채널 순서를 바꾸어준다.
        # 오픈CV의 경우 기본적으로 BGR 순서로 이미지를 읽어온다.
        # 반면에 파이토치의 경우 RBG의 순서로 이미지를 읽어오므로 이 둘이 서로 맞춰주는 것이다.
        img = img.transpose((2, 0, 1))

        # 이제 numpy 배열 형식인 이미지를 파이토치가 인식할 수 있는 텐서로 변환해준다.
        try :
            image = torch.from_numpy(np.flip(img, axis=0).copy()).float()

        # 이미지 읽어오는 시점에서 예외 발생 시, 에러와 이미지 경로 출력
        except Exception as e:
            print(str(e))
            print(self.image_label.iloc[index, 1])

        # 0과 1로 표현되는 라벨도 마찬가지로 파이토치가 인식 가능한 텐서로 변환해준다.
        label = self.image_label.iloc[index, 2]
        label = torch.LongTensor(np.array([label], dtype=np.int64))

        # 변환된 이미지와 라벨 텐서를 하나로 묶어서 반환해준다.
        sample = {'image': image, 'label': label}
        return sample

    def __len__(self):
        return len(self.image_label)

    # 전달 받은 이미지를 무작위로 크기 조절, 각도 변환, 상하좌우 이동, 좌우 반전을 시키는 함수
    # 이를 통해서 신경망은 더 많은 양의 이미지를 학습한 것과 같은 효과를 낸다. (data augmentation)
    def transform_img(self, img, input_size):

        # 입력 이미지로 받을 크기 만큼 이미지의 크기를 조절해준다.
        img=cv2.resize(img, (input_size, input_size))

        # 회전을 줄 각도 범위
        # +- 4도로 각도 변화를 하게 됨
        rotation_range = 4

        # 줌 조절을 할 크기 범위
        # 가로 세로 크기의 0.2만큼 이동하게 됨
        zoom_range = 0.2

        # 상하좌우 이동을 할 범위
        # 가로 세로 크기의 0.05만큼 이동하게 됨
        shift_range = 0.05

        # 좌우 반전을 할 확률
        # 절반의 확률로 좌우가 바뀌게 됨
        random_flip = 0.5

        # 입력 이미지에서 가로 세로 크기를 가져온다.
        h, w = img.shape[0:2]

        # 회전, 크기 조절, 가로세로 이동을 순차적으로 수행한다.
        rotation = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(1 - zoom_range, 1)
        tx = np.random.uniform(-shift_range, shift_range) * w
        ty = np.random.uniform(-shift_range, shift_range) * h
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
        mat[:, 2] += (tx, ty)
        result = cv2.warpAffine(img, mat, (w, h), cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        # 1/2확률로 좌우 반전을 수행한다.
        if np.random.random() < random_flip:
            result = result[:, ::-1]

        # 변화를 마친 이미지를 반환해준다.
        return result




# ===========학습시킬 신경망 클래스 부분===========#

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


# ===========학습에 필요한 객체 생성 및 변수 설정 부분===========#
# 코드 실행시 전달되는 입력 이미지 크기를 처리하기 위한 파서
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    dest="input_size",
                    default="input_size",
                    help="input_size")

args = parser.parse_args()
input_size = int(args.input_size)

if input_size!=64 and input_size!=128 and input_size!=256 :
    print("input_size 값을 64, 128, 256 가운데 하나의 값으로 입력하세요")
    sys.exit()

# 학습을 진행할 신경망 객체를 생성한다.
# 이때 입력 이미지 크기를 설정해준다.
resnet = ResNet(ResidualBlock, [3, 3, 3], input=input_size)

# 신경망 학습 시에 GPU를 사용하기 위한 설정을 해준다.
resnet.cuda()

# 학습의 기준이 되는 비용 함수와 최적화 함수를 설정한다.
# lr은 학습을 진행할 속도에 해당하는 learning rate이다.
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

# 학습과 테스트에 필요한 이미지 경로와 라벨을 담고 있는 csv 파일 경로를 설정한다.
label_dir = './mission1_labels/'
test_label=label_dir+'test_label_sample.csv'
train_label=label_dir+'train_label_sample.csv'

# 학습용, 테스트용 csv 파일에서 이미지와 라벨을 읽어오는 데이터 셋 객체를 생성한다.
# 데이터 셋 객체는 단순 이미지와 라벨을 신경망이 학습 가능한 텐서 형식으로 변환해준다.
train_dataset = CustomDataset(isTrain=True, train_label=train_label, test_label=test_label, input_size=input_size)
test_dataset = CustomDataset(isTrain=False, train_label=train_label, test_label=test_label, input_size=input_size)


# 데이터 셋 객체에서 일정한 묶음 단위로(batch size) 한번에 데이터를 읽어오는 데이터 로더 객체 생성
# 이는 학습을 빠르게 수행할 수 있도록 돕는다.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=50,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10,
                                          shuffle=False)

# ===========학습 진행 부분===========#
for epoch in range(10):
    resnet.train()
    last_loss=0

    # train_loader에서 학습할 이미지와 라벨을 한 묶음 읽어온다.
    for i, sample in enumerate(train_loader):
        images = Variable(sample['image'].cuda())
        labels = Variable(sample['label'].cuda())
        labels = labels.squeeze()

        # 최적화 함수를 초기화 해준다.
        optimizer.zero_grad()

        # 신경망을 앞으로 통과하여 합성일 확률을 계산한다.
        outputs = resnet(images)

        # 이를 라벨과 비교하여 비용을 계산한다.
        loss = criterion(outputs, labels)

        # 이제 신경망을 거꾸로 거슬로 올라가면서 각 단계별로 조절해야할 값을 계산한다.
        loss.backward()

        # 계산한 값만큼 업데이트를 해준다.
        optimizer.step()
        if (i + 1) % 20 == 0:
            print("Epoch : " + str(epoch + 1) + " Iter : " + str((i + 1) * 5) + " loss : " + str(round(loss.data[0], 4)))
            cur_iter = epoch * 1000 + (i + 1) * 5

    # 학습 에포크 수가 길어질 경우 학습률을 서서히 감소시켜서 학습 결과가 수렴하게끔 돕는다.
    if (epoch + 1) % 10 == 0:
        lr /= 1.25
        optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

    # 한 에포크 학습을 마칠 때마다 정확도 테스트를 수행한다.
    # 그 다음 해당 에포크 학습 결과 모델을 저장한다.
    correct = 0
    total = 0
    resnet.eval()

    # 테스트 로더에서 한 묶음의 이미지와 라벨을 읽어온다.
    for sample in test_loader:
        images = Variable(sample['image'].cuda())
        labels =sample['label']
        sq_label = labels.squeeze()

        # 신경망을 앞으로 통과시켜 합성일 확률들을 산출한다.
        outputs = resnet(images)

        # 이를 다시 0과 1로 표현된 라벨로 바꾸어 준다.
        # 그 다음 실제 라벨과 비교하여 얼마나 맞췄는 지를 테스트한다.
        _, predicted = torch.max(outputs.data, 1)
        total += sq_label.size(0)
        correct += (predicted.cpu() == sq_label).sum()

    # 정확도를 출력하고, 모델을 저장한다.
    accuracy=round(100 * (float(correct) / float(total)),2)
    print(str(epoch + 1)+":" + str(accuracy)+"\n")

    model_name = './mission1_checkpoint/mission1_'+ str(input_size) +'_'+str(epoch)+'.pth'
    torch.save(resnet.state_dict(), model_name)


