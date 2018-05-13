# -*- coding: utf-8 -*-

# 작성자 : 이동호

#selenium과 chromedriver가 필요합니다~~
#chromedriver 파일은 구글 드라이브에 올려두었으니 다운받아 활용하시기 바랍니다.
#github에 가시면 모든 spider가 다 있는 형태의 scrapy project 폴더를 만들어놓았습니다~~
import scrapy
from scrapy.selector import Selector
from scrapy.http import Request
from selenium import webdriver
from PIL import Image
from io import BytesIO
import requests
import time
import cv2
import numpy as np

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36")

driver = webdriver.Chrome('chromedriver', chrome_options=options)
driver.implicitly_wait(3)

class Tumblr_Crawler(scrapy.Spider):
    #scrapy crawl tumblr - 할 때 크롤러의 이름을 기재
    name = "tumblr"
    
    #생성자
    def __init__(self):
        self.keyword = 'selfie' # 여기에서 검색 키워드를 변경하시면 됩니다.
        self.path = 'tumblr_'+ self.keyword + '_'
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #request를 보내는 함수
    def start_requests(self):
        url = "https://www.tumblr.com/search/"+ self.keyword
        yield scrapy.Request(url, self.parse)

    #parsing 함수
    def parse(self, response):
        cnt = 0
        driver.get(response.url)
        #100번 동적 페이징
        for i in range(1,100):
            images = driver.find_elements_by_xpath('//img[not(contains(@src,"pixel.quantserve.com") or contains(@src,"avatar_"))]')
            for image in images:
                url = image.get_attribute('src')
                request = Request(url, callback=self.save_photo)
                request.meta['image_num'] = cnt
                cnt = cnt + 1
                yield request
            #브라우저의 끝에 왔을 때
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            #아이템들이 다 로딩될 때까지 조금 쉬고
            time.sleep(5)
        
    def save_photo(self, response):
        
        image_request_result = requests.get(response.url)
        if image_request_result.status_code != 200:
            self.crawler_log.writelines("통신 실패 : " + str(image_request_result.status_code))
            return

        # global file_bytes
        file_bytes = BytesIO(image_request_result.content)
        # print(">> file_bytes = "+str(file_bytes.__sizeof__()))
        # 이미지 데이터의 크기가 0 일 경우 실패 메세지를 띄우고 종료한다.
        if file_bytes.__sizeof__()==0 :
        # print(">> file_bytes = 0 ")
            print("download fail")
            return

        # 크기가 0이 아닌 경우, 사이즈를 확인한다.
        # 지나치게 큰 이미지의 경우는 조절해준다.
        pil_img =Image.open(BytesIO(image_request_result.content))
        width, height = pil_img.size
        max_size = [1024, 1024]
        if width > 200 or height > 200:
            pil_img.thumbnail(max_size)

        # 얼굴 검출을 위해서 PIL 이미지 객체를 opencv 이미지로 변환
        pil_img2 = pil_img.convert('RGB') # opencv이미지는 BGR이라서 맞춰줘야함
        cv_img = np.array(pil_img2)
        cv_img = cv_img[:, :, ::-1].copy()

        # 이미지를 흑백으로 변환한 뒤, 얼굴 검출 진행
        # 흑백인 이유는 얼굴 검출 시에 형태만 활용되므로 색상 정보가 불필요하기 때문
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴이 검출되지 않을 경우 함수를 종료한다.
        if len(faces)==0:
            return

        full_path = self.path + str(response.meta['image_num']) + '.jpg'
        cv2.imwrite(full_path, cv_img)

        
        
        
        
        
        
        