#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/7/27 下午4:01
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import time
import cv2 as cv

from mtcnn.nets.matcnn import MTCNN
from dsfd.detector import FaceDetector


def face_detect(algorithm='mtcnn', image_path=None, save_path=None):
    """

    :param algorithm: mtcnn | dfsd
    :param image_path:
    :param save_path:
    :return:
    """
    image_list = get_file(image_path)
    detector = None
    if algorithm == 'mtcnn':
        detector = MTCNN()
    elif algorithm == 'dsfd':
        detector = FaceDetector()
    else:
         Exception("Only support matcnn and dsfd")

    for image_file in image_list:

        image = cv.imread(image_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        start_time=time.perf_counter()
        end_time = None
        if algorithm == 'mtcnn':
            bboxes = detector.detect_faces(image)
            # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
            # bounding_box = result[0]['box']
            # keypoints = result[0]['keypoints']

            for bbox in bboxes:
                bounding_box = bbox['box']
                confidence = bbox['confidence']
                cv.rectangle(image,  (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0, 255, 255), 4)
                cv.putText(image, str(round(confidence, 2)), (int(bbox[0]), int(bbox[1] - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # cv.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
            # cv.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
            # cv.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
            # cv.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
            # cv.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

        elif algorithm == 'dsfd':
            boxes = detector(image, score_threshold=0.5)
            end_time = time.perf_counter()

            for box_index in range(boxes.shape[0]):
                bbox = boxes[box_index]

                cv.rectangle(image, (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])), (0, 255, 255), 4)
                cv.putText(image, str(round(bbox[4], 2)), (int(bbox[0]), int(bbox[1]- 10) ),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        print('inference cost %.2f second' % (end_time-start_time))
        cv.imwrite(os.path.join(save_path, os.path.basename(image_file)), image)


def get_file(img_dir):
    """

    :param dir:
    :param fileList:
    :return:
    """
    image_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            image_list.append(os.path.join(root, file))
    image_list = [x for x in image_list if 'jpg' in x or 'png' in x]

    return image_list


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    image_path = '/home/alex/Downloads/wider_face/WIDER_val/images/0--Parade'
    save_path = '../outputs'
    face_detect(algorithm='mtcnn', image_path=image_path, save_path=save_path)