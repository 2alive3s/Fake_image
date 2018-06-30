# Fake_image

2018 Fake image challenge를 위한 code

김형준, 김유리, 김해중, 신웅비, 양유준, 이동호, 이정훈, 정수민, 이다솜 supervised by 기계인간


프로젝트 소개
----------

2018 Fake image challenge에서 제시된 두 가지 임무에 대하여 CNN을 활용한 합성 이미지 탐지 모델을 구축
  
* 임무 1 : 인공지능이 생성한 이미지 탐지
  
* 임무 2 : 얼굴 전체, 혹은 일부가 합성된 이미지 탐지
  
본 프로젝트는 챌린지에서 본선 진출 47팀 가운데 최종 (현재 미정) 위를 기록

* 임무 1 AUROC : (현재 미정)
  
* 임무 2 AUROC : (현재 미정)



합성 이미지 탐지 모델 구조
--------------------

### 임무 1

<img src="./mission1_model.png" width="700" height="300"></img>

### 임무 2

<img src="./mission2_model.png" width="800" height="300"></img>


환경 설정
-------
OS : Ubuntu 16.04 LTS

Programming Language : Python 3.6.4

GPU Library : CUDA 9.0

Etc Library : pip install -r requirements.txt 명령어를 통해 설치

학습용 데이터 셋 준비
----------------

#### 원본 이미지 데이터 셋 다운로드

* celebA : https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8

* 직접 수집한 데이터 셋 : https://drive.google.com/open?id=1Z1Xbi0DuDM-NHsCxNKDAQpdgbvvUFHv7

#### 원본 이미지 크롭

* 원본 이미지 데이터 셋에서 얼굴 영역만 크롭한다. 크롭에는 MTCNN 라이브러리를 사용한다.

* 크롭된 원본 얼굴 이미지들은 모델 학습 시에 진짜로 라벨링 되어 사용된다.

* 임무 1용 원본 이미지 크롭

  * /get_result_code/ 폴더로 이동
  
  * python precrop.py -i "celebA 데이터 셋이 저장된 경로" -o "결과 이미지를 저장할 경로" -s 256
  
  
* 임무 2용 원본 이미지 크롭

  * /get_result_code/ 폴더로 이동
  
  * python precrop.py -i "celebA 데이터 셋이 저장된 경로" -o "결과 이미지를 저장할 경로" -s 128

  * python precrop.py -i "직접 수집한 데이터 셋이 저장된 경로" -o "결과 이미지를 저장할 경로" -s 128


#### 합성 이미지 생성

* 임무 1

  * 아래 링크에서 학습된 progressive GAN 모델을 다운받아 /make_dataset_code/make_mission1_data/ 폴더에 넣어준다. 

  * https://drive.google.com/open?id=1Cc2AWAPFEC-CvxJu3__gYKVDYtLZV7a7

  * /make_dataset_code/make_mission1_data/ 폴더로 이동한다.

  * python make_gan_img.py -n "생성할 이미지 수" -s "생성할 이미지 가로 세로 크기" 명령어를 실행한다.

  * 생성된 합성 이미지들은 /dataset/mission1_"생성한 이미지 가로 세로 크기" 폴더에 저장된다.



* 임무 2

  * /make_dataset_code/make_mission2_data/ 폴더로 이동한다.
  
  * python make_faceswap_img.py -t "합성 부위" -f "페더링 적용 여부" -i "크롭된 얼굴 이미지 디렉토리 경로" 실행
  
  * 현재 선택 가능한 부위는 LEFT_BROW, LEFT_EYE, RIGHT_BROW, RIGHT_EYE, EYES, NOSE, MOUTH, INFACE 이다.
  
  * 가장자리에 페더링 적용을 할 시 True 값을 전달하며, 그렇지 않을 경우 False를 전달한다.
  
  * 합성된 이미지는 /dataset/mission2_fake_"합성한 영역 이름" 폴더에 저장된다.

#### 모델 학습용 라벨 파일 생성

* 원본 이미지 데이터 셋과 합성 이미지 데이터 셋의 정보를 하나의 csv 파일에 기록한다.

* csv 파일에는 이미지 경로와 라벨이 기록된다.

* 이 csv 파일을 읽어와서 모델 학습을 진행하게 된다.

* 임무 1 라벨 파일 생성
  
  * /make_dataset_code/make_mission1_data/ 폴더로 이동
  
  * python make_label.py -r "진짜 얼굴 이미지 저장 디렉토리" -f "합성 얼굴 이미지 디렉토리" 명령어 실행
  
  * 생성된 라벨 파일은 /model_train_code/mission1_labels/ 폴더에 저장된다.
  
 * 임무 2 라벨 파일 생성
  
  * /make_dataset_code/make_mission2_data/ 폴더로 이동
  
  * python make_label.py -r "진짜 얼굴 이미지 저장 디렉토리" -f "합성 얼굴 이미지 디렉토리" 명령어 실행
  
  * 생성된 라벨 파일은 /model_train_code/mission1=2_labels/ 폴더에 저장된다.
 
 
 
모델 학습 시키기
------------

* 임무 1 모델 학습

  * /model_train_code/ 폴더로 이동
  
  * python mission1_train.py -i "입력 이미지 크기" -n "모델 번호" 명령어를 실행
  
  * 실제 챌린지에서는 각각 64, 128, 256 크기에 대하여 각각 3개의 모델을 학습 시켜 앙상블을 구성
  
  * 학습된 모델은 /model_train_code/mission1_checkpoint/ 폴더에 저장

* 임무 2 모델 학습

  * /model_train_code/ 폴더로 이동
  
  * python mission2_train.py -p "합성 탐지할 얼굴 영역_가장자리 처리" (ex -p EYES_blur) 명령어를 실행
  
  * 실제 챌린지에서는 총 여섯 부위에 대한 모델을 학습시켜서 사용하였다.
    
  * (EYES_blur, LEFT_EYE_cut, LEFT_EYE_blur, RIGHT_EYE_cut, RIGHT_EYE_blur, MOUTH_blur)
  
  * 학습된 모델은 /model_train_code/mission2_checkpoint/ 폴더에 저장


결과 도출 하기
------------

* 임무 1 결과 도출

  * /get_result_code/ 폴더로 이동
  
  * python mission1_result.py -i "입력 이미지들이 저장된 경로" 명령어 실행
  
  * /result/mission1_result.txt 파일에 결과 저장
  
* 임무 2 결과 도출
  
  * /get_result_code/ 폴더로 이동

  * python precrop.py -i "입력 이미지들이 저장된 경로"
  
  * python mission2_result.py 명령어 실행
  
  * /result/mission2_result.txt 파일에 결과 저장
  
* AUROC 점수 계산하기

  * /AUROC/ 폴더로 이동
  
  * python main.py -p "예측 결과 텍스트 파일 경로" -ㅣ "정답 라벨 텍스트 파일 경로"

Crawler 사용법
------------
Fake_image/crawler/ 경로에서 다음과 같은 명령어를 실행

(Flickr는 API KEY를 기재해주어야 실행이 되어 같은 폴더안에 넣어 놓지 않음)

Google : scrapy crawl google -a keyword=노바 -a dirname=nova

Naver : scrapy crawl naver -a keyword=노바 -a dirname=nova

Tumblr : scrapy crawl tumblr -a keyword=노바 -a dirname=nova

Insta : scrapy crawl insta -a keyword=노바 -a dirname=nova

keyword와 dirname 다른 이유 : 파일 경로에 한글이 사용될 수 없으므로 한글 keyword의 경우 영어로 바꾸어 dirname에 기재
