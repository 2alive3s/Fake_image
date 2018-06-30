# -*- coding: utf-8 -*-
"""
Created on Mon May 14 01:42:36 2018

@author: samsung
"""

# -*- coding: utf-8 -*-
import scrapy
import datetime
import time
import csv
import re
import sys

import urllib

import requests
from PIL import Image
from io import BytesIO

import numpy as np
import cv2
import os


'''

* 사용 방법
1) start_requests 함수의 keyword를 설정해준다.
keyword는 검색하기를 원하는 단어이다.
ex) "문재인"

2) start_requests 함수의 num = str() 안에 추출을 원하는 이미지 개수를 설정해준다.
50개를 입력하면 50장이 검색되어진다.

3) start_requests 함수의 self.face_cascade에는 haarcascade_frontalface_alt.xml 파일이 있는 곳의 경로를 적어준다.

4) set_photo 함수의 file_path 부분을 원하는 대로 설정해준다.

'''

class NewsUrlSpider(scrapy.Spider):

    name = "naver"
    #생성자
    def __init__(self, keyword='', dirname='', **kwargs):
        super().__init__(**kwargs)
        self.root_dir = 'collected_image/naver/'
        self.keyword = keyword # 여기에서 검색 키워드를 변경하시면 됩니다.
        self.dirname = dirname
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        
    #가장 처음 실행되는 수집 함수
    def start_requests(self):
        num = str(30)  #몇개 이미지 뽑기를 원하는지...

        # 얼굴 인식을 위해 필요한 opencv 모듈
        # 구글 드라이브 상에 haarcascade_frontalface_alt.xml 다운 받고 경로 지정
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        # 네이버 이미지 검색 URL 주소
        naver_image_search_url = "https://s.search.naver.com/imagesearch/instant/search.naver?where=image§ion=image&rev=31&ssl=1&res_fr=0&res_to=0&face=0&color=0&ccl=0&ac=1&aq=0&spq=1&query='"+self.keyword+"'&nx_search_query='"+self.keyword+"'&nx_and_query=&nx_sub_query=&nx_search_hlquery=&nx_search_fasquery=&datetype=0&startdate=0&enddate=0&json_type=6&nlu_query={'timelineQuery':'"+self.keyword+"'}&start=0&display="+num+"&_callback=window.__jindo2_callback.__sauImageTabList"
        naver_image_search_url = str(naver_image_search_url)

        request = scrapy.Request(naver_image_search_url , self.url_extractor)
        yield request


    #각 각 이미지 원본 URL을 추출하는 함수.
    def url_extractor(self, response):
        #뉴스의 내용을 추출한다.
        item_content = response.xpath('//*').extract()[0]
        regex = re.compile("originalUrl.*?,")  #원본 이미지 URL 주소 추출
        result = regex.findall(item_content)
        print('*' * 100)

        for image_url in result:
            decode_url = urllib.parse.unquote(image_url)
            #url에서 불필요한 부분 제거
            regex_url = re.sub('originalUrl\\\\\":','',decode_url)
            regex_url = re.sub('\"','',regex_url)
            regex_url = re.sub(',','',regex_url)
            regex_url = re.sub('\\\\','',regex_url)
            print(regex_url)

            request = scrapy.Request(regex_url , self.set_photo)
            yield request



    #원본 URL에서 이미지 추출
    def set_photo(self, response):

        image_url = response.url

        print('*' * 100)
        print(image_url)

        image_request_result = requests.get(image_url)

        if image_request_result.status_code != 200:
            self.crawler_log.writelines("통신 실패 : " + str(image_request_result.status_code))
            return

        file_bytes = BytesIO(image_request_result.content)

        # 이미지 데이터의 크기가 0 일 경우 실패 메세지를 띄우고 종료한다.
        if file_bytes.__sizeof__()==0 :
            print("download fail")
            return

        # 크기가 0이 아닌 경우, 사이즈를 확인한다.
        # 지나치게 큰 이미지의 경우는 조절해준다.
        pil_img =Image.open(BytesIO(image_request_result.content))
        width, height = pil_img.size
        max_size = [1024, 1024]
        if width > 1024 or height > 1024:
            pil_img.thumbnail(max_size)

        # 얼굴 검출을 위해서 opencv 이미지로 변환
        cv_img = np.array(pil_img)
        cv_img = cv_img[:, :, ::-1].copy()

        # 이미지를 흑백으로 변환한 뒤, 얼굴 검출 진행
        # 흑백인 이유는 얼굴 검출 시에 형태만 활용되므로 색상 정보가 불필요하기 때문
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴이 검출되지 않을 경우 함수를 종료한다.
        if len(faces)==0:
            print("face fail")
            return

        # 얼굴이 포함된 이미지를 저장할 폴더명과 파일명을 설정한다.
        file_path = 'collected_image/naver/' + self.dirname + '/'
        full_path = file_path + 'naver_' + self.dirname + '_' + str(time.time()) + '.jpg'

        # 이미지를 저장할 폴더가 없을 경우 생성해준다.
        if not os.path.exists(file_path) :

            os.makedirs(file_path)
            
        print("face ??")
        cv2.imwrite(full_path, cv_img)

