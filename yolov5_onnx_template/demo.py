from jianzhang_det import jianzhangDet
import os, cv2

if __name__ == '__main__':
    dir = "imgs\\imgs_yolov5s"
    for imgname in os.listdir(dir):
        fullname = os.path.join(dir, imgname)
        print(fullname)
        img = cv2.imread(fullname)
        # ########################
        self_change_img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
        # #################
        # print('1')
        jianzhangDet.predict([img])

