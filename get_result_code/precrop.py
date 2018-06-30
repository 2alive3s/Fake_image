import cv2
import numpy as np
import sys, os
from src import detect_faces, show_bboxes
from PIL import Image, ImageOps
import argparse

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    h, w = image.shape[:2]
    dim = (width, height)
    if h > w :
        r = height / float(h)
        dim = (int(w * r), height)

    else :
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    h, w = resized.shape[:2]

    left = int((width - w)/2)
    right = int((width - w)/2)
    top = int((height - h)/2)
    bottom = int((height - h)/2)
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,(0,0,0))

    if image.shape[:2]!=(128,128) :
        image = cv2.resize(image, (128,128))

    return image

# get input dir which passed from cmd lines
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    dest="input_dir",
                    default="input_dir",
                    help="Input directory",
                    required=True)

parser.add_argument('-o', '--output',
                    dest="output_dir",
                    default="../mission2_faces/",
                    help="Input directory")

args = parser.parse_args()
input_dir = args.input_dir

file_array=os.listdir(input_dir)
file_array.sort()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

id_file = open('id_list.txt', 'w')

for file in file_array :
    img_path = input_dir+file
    PIL_img = Image.open(img_path)
    cv2_img = cv2.imread(img_path)

    filename = file.split('.')[0]
    id_file.write(filename+'\n')

    # array that contain fake probability of faces in the img
    total_prob = []

    # detect faces using mtcnn
    bboxs, landmarks = detect_faces(PIL_img)
    org_height, org_width, channel = cv2_img.shape

    # crop face area in the img
    for i in range(len(bboxs)):

        x1, y1, x2, y2, score = bboxs[i]
        height = int(y2) - int(y1)
        width = int(x2) - int(x1)

        # filtering small face img which is unlikely contain fake part
        if height < 50 and width < 50:
            continue

        # filtering area which is not likely face
        if score < 0.97:
            continue

        # crop the face area with some paddings
        plus_rate = 0.3
        height_plus = plus_rate * height
        width_plus = plus_rate * width

        crop_y1 = 0
        crop_y2 = org_height
        crop_x1 = 0
        crop_x2 = org_width

        if y1 - height_plus > 0:
            crop_y1 = int(y1 - height_plus)
        if y2 + height_plus < org_height:
            crop_y2 = int(y2 + height_plus)
        if x1 - width_plus > 0:
            crop_x1 = int(x1 - width_plus)
        if x2 + width_plus < org_width:
            crop_x2 = int(x2 + width_plus)

        crop_img = cv2_img[crop_y1:crop_y2, crop_x1:crop_x2]
        height, width, chennel = crop_img.shape

        # resize the cropped area maintaing the width height ratio
        img = image_resize(crop_img, height=128, width=128)

        output_path=output_dir+filename+'_'+str(i)+'.jpg'
        cv2.imwrite(output_path, img)

        print(output_path)
