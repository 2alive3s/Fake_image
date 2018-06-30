# 생성자 : 김형준
# 최종 수정일 : 18.6.30
# 파일 개요
# 신경망을 학습시키기 위해 team_haechi/dataset 안에 들어있는 이미지 데이터들의 경로와 라벨을
# csv 파일 형식으로 정리하는 파일

import csv
import os
import argparse

parser = argparse.ArgumentParser()
# 코드 실행시 전달되는 파라미터 파싱
# 진짜 이미지들이 담긴 디렉토리 경로
parser.add_argument('-r', '--real_dir',
                    dest="real_dir",
                    default="../../dataset/mission1_real_sample/",
                    help="real img directory")

# 가짜 이미지들이 담긴 디렉토리 경로
parser.add_argument('-f', '--fake_dir',
                    dest="fake_dir",
                    default="../../dataset/mission1_fake_sample/",
                    help="width, height size of output images")

args = parser.parse_args()

# 진짜, 가짜 이미지가 담긴 디렉토리 지정
real_dir = args.real_dir
fake_dir = args.fake_dir

# 해당 디렉토리에서 파일 목록을 가져옴
real_list = os.listdir(real_dir)
real_list.sort()

fake_list = os.listdir(fake_dir)
fake_list.sort()

# 결과 csv 파일을 저장할 디렉토리
output_dir = '../../model_train_code/mission1_labels/'

# 학습용 라벨을 저장할 파일과 테스트용 라벨을 저장할 파일을 생성한다.
train_file = open(output_dir+'train_label_sample.csv','w')
train_csv = csv.writer(train_file)

test_file = open(output_dir+'test_label_sample.csv','w')
test_csv = csv.writer(test_file)

# 학습용 라벨과 테스트 라벨에 각각 컬럼명을 기입해준다.
train_csv.writerow(['seqid', 'img_url', 'label'])
test_csv.writerow(['seqid', 'img_url', 'label'])

train_seq = 0
test_seq = 0
# 가짜 이미지와 진짜이미지를 번갈아가며 라벨 파일에 기입해준다.
# 전체 데이터 셋에서 80%는 훈련용 라벨에 기입해주고, 나머지 20%는 테스트용 라벨에 기입해준다.
for i in range(len(fake_list)) :
    if i < len(real_list)*0.8 :
        train_csv.writerow([train_seq, os.path.abspath(fake_dir + fake_list[i]), 1])
        train_seq+=1
        train_csv.writerow([train_seq, os.path.abspath(real_dir + real_list[i]), 0])
        train_seq += 1

    else :
        test_csv.writerow([test_seq, os.path.abspath(fake_dir + fake_list[i]), 1])
        test_seq += 1
        test_csv.writerow([test_seq, os.path.abspath(real_dir + real_list[i]), 0])
        test_seq += 1


