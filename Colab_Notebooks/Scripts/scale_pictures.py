import cv2
from os import listdir
import numpy as np

"""scale image to percent width and height image"""

imgs_out_path = "C:/Users/Pasca/Downloads/40 B2/"
imgs_path = "C:/Users/Pasca/Downloads/40 B/40 B/"

imgs = listdir(imgs_path)

percent = 30

is_img = True
for img_path in imgs:
    if(is_img):
        img = cv2.imread(imgs_path + img_path, cv2.IMREAD_UNCHANGED)
        print('Original Dimensions : ', img.shape)
        width = int(img.shape[1] *  percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        print('Resized Dimensions : ', resized.shape)

        #cv2.imshow("Resized image", resized)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(imgs_out_path + img_path, resized)
        print(img_path)

    is_img = True #not is_img #for only using images
    #break