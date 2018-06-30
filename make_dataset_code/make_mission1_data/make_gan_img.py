# 생성자 : 김형준
# 최종 수정일 : 18. 6. 30
# 파일 개요 : 미리 학습된 progressive GAN 모델을 읽어와서 설정한 숫자만큼 합성 데이터를 만들어내는 코드
# 필요한 환경 : tensorflow GPU 환경에서 동작

# 실행 방식
# 먼저 다음의 링크에서 progressive GAN pretrained 모델을 다운받은 다음, 이 코드가 속한 디렉터리 안에 위치시킨다.
# 모델 다운 링크 : https://drive.google.com/open?id=1Cc2AWAPFEC-CvxJu3__gYKVDYtLZV7a7
# 모델을 별도로 분리한 이유는 용량상 깃 허브에 업로드 되지 않기 때문
# python make_mission1_data.py -n (이미지 생성 숫자) -s (생성할 이미지 가로세로 크기)

import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import argparse
import os

# 코드 실행 시 전달되는 파라미터 파싱
# 1. 합성 데이터를 저장할 디렉토리 경로
# 2. 만들어 낼 합성 이미지의 수

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--number',
                    dest="make_num",
                    default="1000",
                    help="how many images to make",
                    required=True)

parser.add_argument('-s', '--img_size',
                    dest="img_size",
                    default="64",
                    help="width, height size of output images",
                    required=True)

args = parser.parse_args()
make_num = int(args.make_num)
img_size = int(args.img_size)
output_dir = '../../dataset/mission1_fake_'+str(img_size)+'/'
if not os.path.exists(output_dir) :
    os.mkdir(output_dir)

# 텐서 플로우 세션 초기화
tf.InteractiveSession()

# 미리 학습된 얼굴 생성 GAN 모델 가져오기
with open('./progressive_gan_pretrained.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# 이미지를 생성하기 위해 초기에 입력되는 랜덤 벡터를 생성
# 이를 통해서 각기 다른 얼굴 이미지가 생성된다
latents = np.random.RandomState(make_num).randn(make_num, *Gs.input_shapes[0][1:])

for i in range(0,int(make_num/10)) :

    # 배치 연산을 하기 위해 생성된 랜덤 벡터를 10개 단위로 묶어준다.
    cur_latent = latents[[10*i, 10*i+1, 10*i+2, 10*i+3, 10*i+4, 10*i+5, 10*i+6, 10*i+7, 10*i+8, 10*i+9]] # hand-picked top-10

    # 이미지 생성을 위해서 각 랜덤 벡터에 가짜로 라벨링을 붙여준다.
    # 이를 통해 GAN의 생성망은 이 랜덤 벡터를 진짜 얼굴처럼 보이도록 변형한다.
    labels = np.zeros([cur_latent.shape[0]] + Gs.input_shapes[1][1:])

    # 이제 생성망에 랜덤 벡터와 라벨값을 전달해주어서 10개 단위의 합성 얼굴 이미지를 생성한다.
    images = Gs.run(cur_latent, labels)

    # 생성된 이미지는 아직 텐서플로우가 인식하는 텐서 자료형이다
    # 이를 이미지 파일 형태로 저장 가능한 numpy array 형식으로 변환해준다.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)

    # 생성된 이미지의 크기는 1024x1024 크기로 지나치게 해상도가 높다.
    # 이를 초기에 설정한 크기만큼 조절한다.
    # 생성된 이미지는 최종적으로 dataset 폴더 아래에 mission2_fake_"입력 이미지 크기" 디렉토리 아래에 저장된다.
    for idx in range(images.shape[0]):
        face_img=PIL.Image.fromarray(images[idx], 'RGB')
        face_img.thumbnail([img_size,img_size])
        face_img.save(output_dir+str(10*i+idx+1).zfill(5)+'.jpg')
        print((str(10*i+idx+1).zfill(5)+'.jpg'))


