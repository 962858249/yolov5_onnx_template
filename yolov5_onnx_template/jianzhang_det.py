from onnx_detection.onnx_detector import ONNX_Detector
from params import jianzhang_opt
from onnx_detection.onnx_utils import vis
import cv2
# demo.py
# img = cv2.imread(fullname)
#         jianzhangDet.predict([img])
class JianZhang:
    def __init__(self):
        self.detector = ONNX_Detector(jianzhang_opt)

    def predict(self, images=[]):
        all_results = self.detector.predict(images)
        # print(all_results,'all_results')
        for img, result in zip(images, all_results):
            if result is not None:
                final_boxes, final_cls_inds, final_scores = result
                vis(img, final_boxes, final_scores, final_cls_inds,
                                 conf=jianzhang_opt.score_thr, class_names=jianzhang_opt.class_names)
                print(final_boxes)
                print(final_scores)

            cv2.imshow("vis", img)
            # ##################自己写的，试着改变保存图片大小，但是好像输出和输入的大小是一致的，就不用改了
            # self_change_img = cv2.resize(img, dsize=None, fx=100, fy=100, interpolation=cv2.INTER_LINEAR)
            # ##########################
            # cv2.imwrite("11.jpg",img)
            # cv2.imwrite("11.jpg", self_change_img)
            cv2.waitKey(0)
jianzhangDet = JianZhang()



