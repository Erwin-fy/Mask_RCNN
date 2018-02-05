import os
import numpy as np
import cv2
import re

# Root directory of the project
ROOT_DIR = os.getcwd()


# box_dir = os.path.join(ROOT_DIR, "results", "crop_preserve", "JSRT/800", "renet_C5_wave", "512_320", "box")
# back_dir = os.path.join(ROOT_DIR, "results", "crop_preserve", "JSRT/800", "renet_C5_wave", "512_320", "back")
box_dir = "/media/Disk/wangfuyu/Mask_RCNN/refine/HED/cxr/renet_C5_GRU/512_320/box"
back_dir = "/media/Disk/wangfuyu/Mask_RCNN/refine/HED/cxr/renet_C5_GRU/512_320/back"

'''
mask_results_dir = '/media/Disk/wangfuyu/Mask_RCNN/unet/results/crop'

val_box_info = open('/media/Disk/wangfuyu/Mask_RCNN/crop_results/box_info.txt', 'r')
lines = val_box_info.readlines()
for line in lines:
    tmp = re.split(' ', line)
    filename = tmp[0]
    y1 = int(tmp[1])
    x1 = int(tmp[2])
    y2 = int(tmp[3])
    x2 = int(tmp[4])

    mask = cv2.imread(os.path.join(mask_results_dir, filename + '.png'), cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    result = np.zeros((1024, 1024))
    result[y1: y2, x1: x2] = mask

    cv2.imwrite(os.path.join(temp_dir, filename + '.png'), result,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
'''

isExists = os.path.exists(back_dir)
if not isExists:
    os.makedirs(back_dir)
val_id = open('/media/Disk/wangfuyu/Mask_RCNN/data/cxr/val_id.txt')
# val_id = open('/media/Disk/wangfuyu/Mask_RCNN/data/cxr/800/JSRT/even_id.txt')
lines = val_id.readlines()
for line in lines:
    # print (line + '-')
    result = np.zeros((1024, 1024), dtype=np.int64)

    for index in range(0, 3):
        filename = line[0:-1] + '_' + str(index)
        print (filename)
        temp = cv2.imread(os.path.join(box_dir, filename + '.png'), cv2.COLOR_BGR2GRAY)
        if temp is not None:
            temp = temp.astype(np.int64)
            print ("Not")
            result = np.bitwise_or(result, temp)

    result = result.astype(np.uint8)
    cv2.imwrite(os.path.join(back_dir, line[0:-1] + '.png'), result,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 9])