from skimage import io,color
from PIL import Image
import os
import numpy as np

imgroot = "/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/1024/masks/"
jpgroot = "/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/1024/masks/"
txt = open('/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/odd_id.txt', 'r')
aug_txt = open('/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/odd_id_aug.txt', 'w')

for line in txt.readlines():
    aug_txt.write(line[0:-1] + '\n')
    aug_txt.write(line[0:-1] + '_p3' + '\n')
    aug_txt.write(line[0:-1] + '_m3' + '\n')
    aug_txt.write(line[0:-1] + '_p5' + '\n')
    aug_txt.write(line[0:-1] + '_m5' + '\n')
    aug_txt.write(line[0:-1] + '_p8' + '\n')
    aug_txt.write(line[0:-1] + '_m8' + '\n')

# for filename in os.listdir(imgroot):
    # txt.write(filename[0:-4] + '\n')
    # print (filename)
    #
    # infile = Image.open(imgroot+filename)
    # img = np.array(infile, dtype=np.float32)
    # img = img / 255.0
    #
    # if img is None:
    #     continue
    #
    # # +3
    # hsvimg = color.rgb2hsv(img)
    # hsvimg[:,:,2] += 3.0/255.0
    # hsvimg[hsvimg[:,:,2]>255] = 255
    # aug = color.hsv2rgb(hsvimg)
    # augimg = Image.fromarray(255*aug[:,:,0]).convert('RGB')
    # augimg.save(jpgroot+filename[0:-4]+'_p3.jpg')
    #
    # # -3
    # hsvimg = color.rgb2hsv(img)
    # hsvimg[:, :, 2] -= 3.0 / 255.0
    # hsvimg[hsvimg[:, :, 2] > 255] = 255
    # aug = color.hsv2rgb(hsvimg)
    # augimg = Image.fromarray(255 * aug[:, :, 0]).convert('RGB')
    # augimg.save(jpgroot + filename[0:-4] + '_m3.jpg')
    #
    # # +5
    # hsvimg = color.rgb2hsv(img)
    # hsvimg[:, :, 2] += 5.0 / 255.0
    # hsvimg[hsvimg[:, :, 2] > 255] = 255
    # aug = color.hsv2rgb(hsvimg)
    # augimg = Image.fromarray(255 * aug[:, :, 0]).convert('RGB')
    # augimg.save(jpgroot + filename[0:-4] + '_p5.jpg')
    #
    # # -5
    # hsvimg = color.rgb2hsv(img)
    # hsvimg[:, :, 2] -= 5.0 / 255.0
    # hsvimg[hsvimg[:, :, 2] > 255] = 255
    # aug = color.hsv2rgb(hsvimg)
    # augimg = Image.fromarray(255 * aug[:, :, 0]).convert('RGB')
    # augimg.save(jpgroot + filename[0:-4] + '_m5.jpg')
    #
    # # +8
    # hsvimg = color.rgb2hsv(img)
    # hsvimg[:, :, 2] += 8.0 / 255.0
    # hsvimg[hsvimg[:, :, 2] > 255] = 255
    # aug = color.hsv2rgb(hsvimg)
    # augimg = Image.fromarray(255 * aug[:, :, 0]).convert('RGB')
    # augimg.save(jpgroot + filename[0:-4] + '_p8.jpg')
    #
    # # -8
    # hsvimg = color.rgb2hsv(img)
    # hsvimg[:, :, 2] -= 8.0 / 255.0
    # hsvimg[hsvimg[:, :, 2] > 255] = 255
    # aug = color.hsv2rgb(hsvimg)
    # augimg = Image.fromarray(255 * aug[:, :, 0]).convert('RGB')
    # augimg.save(jpgroot + filename[0:-4] + '_m8.jpg')