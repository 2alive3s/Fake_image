import scrapy
from scrapy.spiders import BaseSpider
import time, cv2, os, json
from io import BytesIO
from PIL import Image
import numpy as np
import requests
from urllib.parse import unquote


# 작성자 : 김형준
# 최종 수정일 : 18. 6. 30.
# 파일 개요 : 인스타그램에서 특정 해쉬 태그로 검색한 결과 이미지들을 수집하여 저장하는 크롤러

# 셋팅
# init 함수 내의 변수들의 값만 자신의 환경에 맞게끔 변경해주고 실행시키면 된다.
# 설정해야하는 변수 : root_dir, face_cascade, dirname_dict 이다.
# 변수별 상세 내용은 각각의 바로 위 주석을 참고할 것
# 멀티 쓰레드 설정은 setting.py의 CONCURRENT_REQUESTS 값을 변경하면 된다.
# 지나친 크롤링 요청은 IP 차단의 위험성이 있으므로 3 쓰레드 정도를 권장한다.

class InstaSpider(BaseSpider):
    name = "insta"

    def __init__(self):
        super(InstaSpider, self).__init__()
        self.start_urls = [
            'https://www.instagram.com/'
        ]
        # 이미지를 저장할 기본 폴더, 자기 환경에 맞게끔 변경할 것
        self.root_dir = '/Users/junny/Desktop/crawler/insta/'

        # 얼굴 검출 시에 사용할 OpenCV 모듈 설정
        # 구글 드라이브 상의 haarcascade_frontalface_alt.xml 파일을 다운로드 받고 경로 지정할 것
        self.face_cascade = cv2.CascadeClassifier(
            '/Users/junny/anaconda3/lib/python3.6/haarcascade_frontalface_alt.xml')

        # 검색 키워드와 저장 폴더명을 설정하는 딕셔너리
        # 형식은 다음과 같다. {'키워드1':'저장 폴더명1', '키워드2':'저장 폴더명2'...}

        # 키워드 네이밍 규칙
        # 한글, 영어 모두 가능, 공백은 포함되어선 안된다.(인스타그램 특성 때문)

        # 디렉토리 네이밍 규칙
        # 영어만 가능
        # 영어 키워드는 그대로 사용
        # 한글 키워드일 시에는 영어로 변환한 뒤 +han 붙여줄 것 (ex. 다비치 -> davichi+han)
        self.dirname_dict = {'연우': 'yeonwoo', '지드래곤': 'gdragon+han', 'justinbiber': 'justinbiber'}

        # 키워드 리스트를 활용하여 시작 url들을 설정해준다.
        # 이 같이 설정한 이유는 여러 키워드들에 대한 크롤링을 동시에 진행하기 위함
        self.start_urls = []
        for keys in self.dirname_dict:
            self.start_urls.append('https://www.instagram.com/explore/tags/' + keys + '/?__a=1')

    def parse(self, response):
        # 현재 키워드 값을 받아온다. 추가 페이지 요청시 필요하기 때문
        keyword = unquote(response.url.split('/')[5])
        dirname = self.dirname_dict[keyword]

        # 반환된 JSON 데이터를 파싱한다.
        graphql = json.loads(response.text)
        hashtag_to_media = graphql['graphql']['hashtag']['edge_hashtag_to_media']
        has_next = hashtag_to_media['page_info']['has_next_page']
        end_cursor = hashtag_to_media['page_info']['end_cursor']
        edges = hashtag_to_media['edges']

        # 각 아이템 별로 가장 고화질인 640x640 이미지의 url을 추출한다.
        for i in range(len(edges)):
            self.set_photo(edges[i]['node']['thumbnail_resources'][4]['src'], dirname)

        # 추가 페이지가 있는지 여부를 확인한다.
        # 그리고 해당 검색어로 저장된 이미지가 1000개 이하일 경우 크롤링을 추가 진행한다.
        if has_next and len(os.listdir(self.root_dir + dirname)) < 50:
            new_url = "https://www.instagram.com/explore/tags/" + keyword + "/?__a=1&max_id=" + end_cursor
            mRequset = scrapy.Request(new_url, callback=self.parse)
            yield mRequset

    def set_photo(self, url, dirname):
        # 전달받은 url을 요청하여 이미지 데이터를 다운받는다.
        image_request_result = requests.get(url)

        # HTTP 통신이 성공했는지 여부를 확인하며, 실패시 종료한다.
        if image_request_result.status_code != 200:
            print("Network Fail")
            return

        file_bytes = BytesIO(image_request_result.content)

        # 이미지 데이터의 크기가 0 일 경우 종료한다.
        if file_bytes.__sizeof__() == 0:
            print("Download Fail")
            return

        # 크기가 0이 아닌 경우, 사이즈를 확인한다.
        # 지나치게 가로 세로 크가기 큰 이미지는 조절해준다. (1024x1024 이상)
        # 크기 조절의 이유를 추가할 것
        pil_img = Image.open(BytesIO(image_request_result.content))
        width, height = pil_img.size
        max_size = [1024, 1024]
        if width > 1024 or height > 1024:
            pil_img.thumbnail(max_size)

        try:
            # 얼굴 검출을 위해서 opencv 이미지로 변환
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except:
            print("변환 실패")
            return

        # 이미지를 흑백으로 변환한 뒤, 얼굴 검출 진행
        # 흑백인 이유는 얼굴 검출 시에 형태만 활용되므로 색상 정보가 불필요하기 때문
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴이 검출되지 않을 경우 함수를 종료한다.
        if len(faces) == 0:
            return

        # 얼굴이 포함된 이미지를 저장할 폴더명과 파일명을 설정한다.
        file_path = self.root_dir + dirname + '/'
        full_path = file_path + 'insta_' + dirname + '_' + str(time.time()) + '.jpg'

        # 이미지를 저장할 폴더가 없을 경우 생성해준다.
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        # 이미지를 지정된 경로에 저장한다.
        cv2.imwrite(full_path, cv_img)
