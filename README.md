# Fake_image

Fake image challenge를 위한 code들입니다.

김형준, 김유리, 김해중, 신웅비, 양유준, 이동호, 이정훈 supervised by 기계인간

--------------------------------------------------------------------------------------------------------------
Crawler 사용법

Fake_image/crawler/ 경로에서 다음과 같은 명령어를 실행

(Flickr는 API KEY를 기재해주어야 실행이 되어 같은 폴더안에 넣어 놓지 않음, Instagram은 개발 중)

Google : scrapy crawl google -a keyword=노바 dirname=nova

Naver : scrapy crawl naver -a keyword=노바 dirname=nova

Tumblr : scrapy crawl tumblr -a keyword=노바 dirname=nova

keyword와 dirname 다른 이유 : 파일 경로에는 영어가 사용될 수 없으므로 keyword를 한글로 작성 시, 이를 영어로 바꾸어 dirname에 기재해야 함


