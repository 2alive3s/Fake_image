# -*- coding: utf-8 -*-
"""
Created on Mon May 14 01:40:14 2018

@author: samsung
"""

import scrapy
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import time
import cv2
import os

# 작성자 : 김형준
# 파일 개요 : 구글에서 특정 키워드 검색 결과 이미지들을 가져와 저장하는 크롤러
            # 이미지를 수집하는 이유는 학습 데이터 셋을 구성하기 위함

# 셋팅
# __init__ 함수 내의 변수들을 자신의 환경에 맞게끔 변경한 뒤 실행할 것
# 구글 드라이브 상에 haarcascade_frontalface_alt.xml 다운 받은 다음 (어디에 저장해도 무방)
# 아래 __init__ 함수 안에 face_cascade 부분에 저장한 경로를 입력해준다.
# 실행 명령어는 별도 파라미터 없이 scrapy crawl GoogleSpider 이다

# 동작 순서는 다음과 같다 - 이건 내가 보기 위한 부분, 다른 사람은 굳이 보지 않아도 된다
# 0. 생성자 함수가 실행되며 사용자가 설정해놓은 이미지 저장 경로, 검색 키워드를 가져온다.
# 1. 구글에 특정 키워드의 검색 결과를 페이지 별로 요청한다.
# 2. 검색 결과 페이지에 포함된 이미지들의 상세보기 페이지로 이동한다.
# 3. 이미지 상세 보기 페이지에 포함된 원본 이미지 URL을 set_photo 함수에 전달한다.
# 4. set_photo 에서는 먼저 이미지 안에 얼굴이 포함되어 있는지 여부를 확인한다.
# 5. 얼굴이 있을 경우, 이미지 파일을 지정된 경로에 저장한다.

class GoogleSpider(scrapy.Spider):
    name = "google"
    def __init__(self, keyword='', dirname='',**kwargs):
        # 이미지를 저장할 기본 폴더, 자기 환경에 맞게끔 변경할 것
        self.root_dir = 'collected_image/google/'

        # 구글 검색창에 입력할 키워드이며 한글과 영어 모두 가능
        # , 로 구분해서 여러 키워드 설정 가능
        self.keyword_list=[keyword]

        # 키워드 별로 이미지 저장할 디렉토리 명이며 영어여야 한다.
        # 특정 키워드 검색 결과 이미지를 저장할 폴더명을 설정하는 배열
        # 디렉토리 명 설정 시 영어 키워드는 그대로 사용
        # 한글 키워드일 시에는 영어로 변환한 뒤 +han 붙여줄 것 (ex. 다비치 -> davichi+han)
        # 키워드에 공백을 포함할 경우 + 로 치환해 줄 것 (ex. 커플 여행 -> couple+travel+han)
        self.dirname_list=[dirname]

        # 얼굴 인식을 위해 필요한 opencv 모듈
        # 구글 드라이브 상에 haarcascade_frontalface_alt.xml 다운 받고 경로 지정
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        # 구글 이미지 검색 기본 URL
        self.search_url='https://www.google.co.kr/search?ei=1rPmWuqdJMTA0gTt94WIBA&yv=3&newwindow=1&tbm=isch&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc&vet=10ahUKEwjq4PTsq-HaAhVEoJQKHe17AUEQuT0INSgB.1rPmWuqdJMTA0gTt94WIBA.i&ved=0ahUKEwjq4PTsq-HaAhVEoJQKHe17AUEQuT0INSgB'

        # 크롤링 시작 URL, 구글 기본 페이지로 설정
        self.start_urls = [
            'https://www.google.com'
        ]

        # 이미지 검색 결과를 가져올 페이지 수
        self.max_page=10

    # 키워드 별로 총 10개의 이미지 검색 결과 페이지를 요청한다.
    def parse(self, response):
        for i in range(0, len(self.keyword_list)) :
            for j in range(0, self.max_page) :
                new_url=self.search_url+'&q='+self.keyword_list[i]+'&ijn='+str(j)+'&start='+str(j*100)
                mRequest=scrapy.Request(new_url, callback=self.parse_imglist)
                mRequest.meta['dirname']=self.dirname_list[i]
                yield mRequest

    # 이미지를 담은 목록 페이지를 파싱
    # 각 이미지별 상세보기 페이지로 넘어간다
    # 상세 보기 페이지로 넘어가는 이유는 썸네일이 아닌, 원본 이미지 획득을 위해서이다.
    def parse_imglist(self, response):
        total_soup = BeautifulSoup(response.text, 'lxml')
        result_area = total_soup.find_all('a', attrs={'jsname':'hSRGPd'}, href=True)
        thumnail_url = total_soup.find_all('img', src=True)

        for i in range(0, len(result_area)) :
            mRequest=scrapy.Request(result_area[i]['href'], callback=self.parse_detail)
            mRequest.meta['dirname']=response.meta['dirname']
            mRequest.meta['thumnail']=thumnail_url[i]['src']
            yield mRequest

    # 이미지 상세보기 페이지 파싱
    # 여기서는 원본 이미지의 링크를 추출하여 set_photo 함수에 전달한다.
    def parse_detail(self, response):
        dirname=response.meta['dirname']
        cur_thumnail=response.meta['thumnail']

        total_soup = BeautifulSoup(response.text, 'lxml')
        img_list = total_soup.find_all('img', src=True)
        if len(img_list)>0 :
            img_url=img_list[1]['src']
            self.set_photo(img_url, dirname)

        else :
            self.set_photo(cur_thumnail, dirname)

    # 이미지 url을 전달받아서 사람 얼굴을 포함하는지 확인한 뒤, 저장하는 함수
    # 얼굴 검출 소스 코드는 아래 주소를 참고
    # https://realpython.com/face-recognition-with-python
    def set_photo(self, url, dirname):

        # 전달받은 url을 요청하여 이미지 데이터를 다운받는다.
        image_request_result = requests.get(url)

        # HTTP 통신이 성공했는지 여부를 확인하며, 실패시 상태 코드와 url을 로그로 남기고 종료한다.
        if image_request_result.status_code != 200:
            self.crawler_log.writelines("통신 실패 : " + str(image_request_result.status_code)+" , "+url)
            return

        file_bytes=BytesIO(image_request_result.content)

        # 이미지 데이터의 크기가 0 일 경우 url을 로그로 남기고 종료한다.
        if file_bytes.__sizeof__()==0 :
            self.crawler_log.writelines("이미지 다운 실패 : " + url)
            return

        # 크기가 0이 아닌 경우, 사이즈를 확인한다.
        # 지나치게 가로 세로 크가기 큰 이미지는 조절해준다. (1024x1024 이상)
        # 크기 조절의 이유를 추가할 것
        pil_img =Image.open(BytesIO(image_request_result.content))
        width, height = pil_img.size
        max_size = [1024, 1024]
        if width > 1024 or height > 1024:
            pil_img.thumbnail(max_size)

        # 얼굴 검출을 위해서 opencv 이미지로 변환
        # cv_img = np.array(pil_img)
        # cv_img = cv_img[:, :, ::-1].copy()

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 이미지를 흑백으로 변환한 뒤, 얼굴 검출 진행
        # 흑백인 이유는 얼굴 검출 시에 형태만 활용되므로 색상 정보가 불필요하기 때문
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴이 검출되지 않을 경우 함수를 종료한다.
        if len(faces)==0:
            return

        # 얼굴이 포함된 이미지를 저장할 폴더명과 파일명을 설정한다.
        file_path = self.root_dir + dirname + '/'
        full_path = file_path + 'google_' + dirname + '_' + str(time.time()) + '.jpg'

        # 이미지를 저장할 폴더가 없을 경우 생성해준다.
        if not os.path.exists(file_path) :
            os.makedirs(file_path)

        # 이미지를 지정된 경로에 저장한다.
        cv2.imwrite(full_path, cv_img)

