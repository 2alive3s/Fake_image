# 생성자 : 김형준
# 최종 수정일 : 18.6.30
# 파일 개요 : 미리 설정한 수만큼 얼굴 일부분이 뒤바뀐 합성 이미지를 만들어 내는 파일

# 샘플 데이터 셋 생성 방법
# python make_faceswap_img.py -t "합성 부위" -f "페더링 적용 여부"
# 용량 관계로 실제 데이터 셋 중 일부를 team_haechi/dataset 폴더 안에 올려놓았다.
# 합성을 완료하면 team_haechi/dataset/mission2_fake_"합성부위" 폴더 안에 저장되어 있다.
# 미리 생성한 합성 이미지 예시는 team_haechi/dataset/mission2_fake_"합성부위"_sample 폴더 안에 저장되어 있다.

# 실제 챌린지에 사용된 데이터 셋 생성 방법
# python make_faceswap_img.py -t "합성 부위" -f "페더링 적용 여부" -i "celebA 얼굴 크롭 디렉토리 경로"
# 실제 챌린지에서는 celebA 데이터 셋에서 얼굴 부분만 크롭한 이미지를 합성에 필요한 얼굴과 몸통 이미지로 사용하였다.
# 1. 다음의 링크에서 celebA 데이터 셋을 다운받는다. https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
# 2. /team_haechi/get_result_code/precrop.py 파일을 실행 시켜서 celebA 데이터 셋에서 얼굴 영역만 크롭한 이미지를 생성한다.
#    자세한 사항은 /team_haechi/get_result_code/precrop.py 주석을 참조할 것
# 3. 얼굴 이미지만 저장한 디렉토리를 코드 실행시 -i 파라미터에 전달하여 실행시키면 된다.

import cv2
import dlib, face_recognition_models
import numpy
import sys
import os
import random
import argparse

# 입력 이미지에서 얼굴이 포함되어 있는지 여부를 검사하는 함수
# 얼굴이 포함되어 있지 않다면 False를 반환, 그 이외의 경우엔 True를 반환
def contain_face(im) :
    im=cv2.imread(im)
    rects = detector(im, 1)
    if len(rects)==0 :
        return False
    else :
        return True

# 얼굴 안에서 68개의 좌표 지점을 추출하는 코드
def get_landmarks(im):
    rects = detector(im, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# 합성할 원본에서 합성될 영역을 검은색으로 지워주는 함수
# 에를 들어 얼굴이 합성될 이미지라면 얼굴 영역을 검은색으로 칠해준다.
# 그 위에 덮어씌울 이미지가 포개 지면서 합성되는 것
def get_face_mask(im, landmarks, feather):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    # 합성하길 원하는 영역의 좌표들을 하나로 이어서 그 안쪽 영역을 검은색으로 칠해주는 함수
    for group in OVERLAY_POINTS:
        points = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(im, points, color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    # 페더링이 설정되어 있다면 가장자리 번짐 처리를 해준다.
    if feather :
        im = (cv2.GaussianBlur(im, (11, 11), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (11, 11), 0)

    # 완성된 마스크를 반환해준다.
    return im

# 자연스러운 합성을 위해 합성할 얼굴의 주요 부위 위치를 원본 얼굴 주요 부위 위치에 맞게 회전 시켜주는 함수
def transformation_from_points(points1, points2):

    # 정수형으로 표현된 각 얼굴별 좌표를 소수형으로 변환
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    # 각 좌표 별로 평균 값을 빼주어서 정규화를 진행한다
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    # 각 얼굴 별로 좌표들의 표준 편차를 구한다.
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)

    # 다시 좌표들을 이 표준 편차로 나누어 주어 정규화한다.
    points1 /= s1
    points2 /= s2

    # 정규화한 좌표들이 서로 일치하도록 회전 각도를 계산한다.
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

# 파일 경로에 해당하는 이미지를 읽어오고 얼굴 좌표를 추출하는 함수
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (int(im.shape[1]), int(im.shape[0] )))
    s = get_landmarks(im)
    return im, s

# 이미지를 지정한 모양에 맞게 휘어주는 함수
# 합성하고자 하는 얼굴 이미지의 각도를 원본 얼굴 이미지의 각도에 맞게 회전 시켜주는 역할을 한다.
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im


def make_half_org(org, half):
    output=np.zeros(org.shape)
    for i in range(len(half)) :
        for j in range(len(half[0])) :
            if not np.array_equal(half[i][j], [0, 0, 0]):
                output[i][j]=org[i][j]
            else :
                output[i][j] = [0,0,0]
    return output


#================얼굴 합성에 필요한 기본 변수 설정 부분===================#

# dlib를 통해 추출한 얼굴 영역 좌표들이 각각 어느 부위에 해당하는 지를 설정한 변수
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
INFACE_POINTS = list(range(0, 27))

# 각 부위별 좌표를 묶어서 양쪽 눈, 왼쪽 눈, 오른쪽 눈, 눈썹 등의 부위를 설정
EYES = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS
LEFT_EYE = LEFT_BROW_POINTS + LEFT_EYE_POINTS
RIGHT_EYE = RIGHT_EYE_POINTS + RIGHT_BROW_POINTS
BROWS = LEFT_BROW_POINTS + RIGHT_BROW_POINTS



# 얼굴 이미지에서 68개의 좌표를 추출하기 위해 필요한 dlib 객체들을 생성
detector = dlib.get_frontal_face_detector()
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
predictor = dlib.shape_predictor(predictor_68_point_model)


#================얼굴 합성 진행 부분===================#

# 코드 실행시 전달되는 파라미터 파싱
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target',
                    dest="target",
                    default="INFACE",
                    help="Which part to swap",
                    required=True)

parser.add_argument('-f', '--feather',
                    dest="feather",
                    default="True",
                    help="Apply feathering to output images",
                    required=True)

parser.add_argument('-i', '--input_dir',
                    dest="input_dir",
                    default="../../dataset/mission2_real_sample/",
                    help="body img directory for faceswap")

args = parser.parse_args()
target = args.target
feather = args.feather

face_dir = args.input_dir
body_dir = args.input_dir

# 얼굴로 사용될 이미지 파일 목록과 몸통으로 사용될 이미지 목록을 가져온다.
# 다양한 합성 이미지 조합을 위해서 얼굴 이미지 파일 목록 순서를 뒤섞어준다.
face_list = os.listdir(face_dir)
random.shuffle(face_list)

body_list = os.listdir(body_dir)
body_list.sort()

# 사용자가 변환하려 하는 영역 좌표를 저장하는 배열
ALIGN_POINTS = []
OVERLAY_POINTS = []

# 사용자가 전달한 좌표에 따라서 합성하고자 하는 얼굴 좌표 설정
target_array = target
if target == "LEFT_BROW":
    ALIGN_POINTS += LEFT_BROW_POINTS

elif target == "LEFT_EYE":
    ALIGN_POINTS += (LEFT_BROW_POINTS + LEFT_EYE_POINTS)

elif target == "RIGHT_EYE":
    ALIGN_POINTS += (RIGHT_BROW_POINTS + RIGHT_EYE_POINTS)

elif target == "RIGHT_BROW":
    ALIGN_POINTS += RIGHT_BROW_POINTS

elif target == "LEFT_EYE":
    ALIGN_POINTS += LEFT_EYE_POINTS

elif target == "RIGHT_EYE":
    ALIGN_POINTS += RIGHT_EYE_POINTS

elif target == "EYES":
    ALIGN_POINTS += EYES

elif target == "NOSE":
    ALIGN_POINTS += NOSE_POINTS

elif target == "MOUTH":
    ALIGN_POINTS += MOUTH_POINTS

elif target == "INFACE":
    ALIGN_POINTS = INFACE_POINTS

OVERLAY_POINTS = [ALIGN_POINTS]

# 합성하는 영역과 페더링 적용 여부에 따라서 결과 이미지 저장 경로가 달라짐
if feather =="True":
    output_dir = '../../dataset/mission2_fake_'+target+'_blur/'
else :
    output_dir = '../../dataset/mission2_fake_'+target+'_cut/'

if not os.path.exists(output_dir) :
    os.mkdir(output_dir)

# 몸통으로 사용할 이미지 개수만큼 반복하며 합성 이미지를 생성한다.
for i in range(len(body_list)) :
    # 1000개 합성시마다
    if (i+1)%1000==0 :
        print((i+1),' 진행 중')

    # 현재 합성에 사용할 얼굴과 몸통 이미지 경로 지정
    cur_body = body_dir + body_list[i]
    cur_face = face_dir + face_list[i]

    # 몸통이나 얼굴용 이미지에서 얼굴이 검출 안될 경우 멈춤
    if not contain_face(cur_body) or not contain_face(cur_face):
        continue

    # 얼굴과 몸통 이미지를 읽어와 얼굴 영역 좌표를 추출한다.
    im1, landmarks1 = read_im_and_landmarks(cur_body)
    im2, landmarks2 = read_im_and_landmarks(cur_face)

    # 자연스러운 합성을 위해서는 합성 시킬 얼굴 이미지가 원본 이미지의 얼굴과 각도가 일치해야한다.
    # 이를 위해 회전 시킬 각도를 나타내는 M 행렬을 계산한다.
    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    # 합성 시킬 이미지에서 합성시킬 영역만 잘라오는데 사용될 마스크를 생성한다.
    mask = get_face_mask(im2, landmarks2, feather)

    # 이제 이 마스크를 원본 이미지 얼굴 각도에 맞게 회전시켜준다.
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1, feather), warped_mask], axis=0)

    # 이제 합성시킬 이미지를 원본 이미지 얼굴 위치에 일치하도록 회전시켜준다
    warped_im2 = warp_im(im2, M, im1.shape)

    # 회전 시킨 이미지에서 합성 시킬 영역만 잘라낸다
    overlay_img = warped_im2 * combined_mask

    # 원본 이미지에서 합성 시킬 영역에 해당하는 부분을 지워준다.
    # 예를 들어 양쪽 눈을 합성할 경우, 그 영역을 검은색으로 칠해주는 작업이다.
    background_img = im1 * (1.0 - combined_mask)

    # 이제 합성시킬 영역이 지워진 원본 이미지와, 합성 시킬 영역만 잘라낸 이미지를 합친다.
    output_im = background_img + overlay_img

    # 결과 이미지를 지정된 경로에 저장한다.
    output_path = output_dir + str(i).zfill(5)+'.jpg'
    cv2.imwrite(output_path, output_im)

