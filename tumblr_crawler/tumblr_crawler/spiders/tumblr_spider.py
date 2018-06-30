# -*- coding: utf-8 -*-
"""
Created on Fri May  4 02:10:38 2018

@author: samsung
"""

import scrapy

from scrapy.selector import Selector
from scrapy.http import Request
from selenium import webdriver
from PIL import Image
from io import BytesIO
import requests
import time

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36")

driver = webdriver.Chrome('chromedriver', chrome_options=options)
driver.implicitly_wait(3)

class Tumblr_Crawler(scrapy.Spider):
    name = "tumblr"

    def start_requests(self):
        url = "https://www.tumblr.com/search/"+'selfie' #검색어
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        cnt = 0
        driver.get(response.url)
        for i in range(1,100):
            images = driver.find_elements_by_xpath('//img[not(contains(@src,"pixel.quantserve.com") or contains(@src,"avatar_"))]')
            for image in images:
                url = image.get_attribute('src')
                request = Request(url, callback=self.save_photo)
                request.meta['image_num'] = cnt
                cnt = cnt + 1
                yield request
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
        
    def save_photo(self, response):
        response_url = requests.get(response.url)
        image = Image.open(BytesIO(response_url.content))
        filepath = 'tumblr_'+'selfie_'+ str(response.meta['image_num'])+'.jpg' #검색어 + 일련번호 적어주기
        image.save(filepath)
        
        
        
        
        
        