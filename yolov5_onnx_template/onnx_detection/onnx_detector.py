import onnxruntime
import numpy as np
import cv2
from .onnx_utils import preproc as preprocess
from .onnx_utils import mkdir, multiclass_nms, letterbox, vis, scale_coords

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNX_Detector:
    def __init__(self, opt):
        self.opt = opt
        onnx_file, imgsz, device, score_thr = opt.model_file, opt.img_size, opt.device, opt.score_thr

        logger.info("....initialize....")
        self.session = onnxruntime.InferenceSession(onnx_file)
        # 创建一个InferenceSession的实例并传给它一个模型地址
        self.input_shape = (imgsz, imgsz)
        self.score_thr = score_thr
        if device<0:#jianzhang.opt.device=-1
            self.session.set_providers(['CPUExecutionProvider'])
        #调用run方法进行模型推理。因此onnxruntime模块中的InferenceSession就是我们的切入点。

        # 进行一次前向推理,测试程序是否正常  向量维度（1，3，imgsz，imgsz）
        img = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)  # init img

        # #############################
        print(self.session.get_inputs()[0].name,'2',img.shape,'3','!!!!!!!!!!!!!!!!!!')
        print(self.session.get_outputs()[0].name,'!!!!!!!!!!!!!!!!')

        output = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})

    def preprocess(self, img0):
        img = letterbox(img0, self.input_shape, stride=self.input_shape[0])[0]#将标记物体放缩和填充
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB  [：，：，：：-1]的作用就是把RGB(或BRG)转换成BGR(或者RGB)。
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    def predict(self, images=[], score_thr=-1):
        if not isinstance(images, list):
            raise TypeError("The input data is inconsistent with expectations.")
        if score_thr <0 or score_thr >1:
            score_thr = self.score_thr


        all_results = []
        for img0 in images:
            if img0 is None or not isinstance(img0, np.ndarray) or not len(img0.shape) == 3:
                logger.info("error image")
                all_results.append(None)
                continue
            print(type(img0))
            img = self.preprocess(img0)
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            output = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})
            # predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]
            predictions = output[0]# [1 258 8]
            predictions = predictions.reshape((predictions.shape[1], predictions.shape[2]))#258 8
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.

            #x y w h  ,= f(x1,y1,x2,y2),后者转换成前者的格式

            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            # boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)


            # Process predictions
            # dets[:, 0:4] = scale_coords(img.shape[2:], dets[:, 0:4], origin_img.shape).round()

            if dets is not None:
                dets[:, 0:4] = scale_coords(img.shape[2:], dets[:, 0:4], img0.shape).round()

                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                ids = final_scores > score_thr
                #找出概率大于设定值的final_box 最终显示在图片上 [ True  True  True  True False False False False  True  True False False, False]
                final_boxes = final_boxes[ids]
                final_scores = final_scores[ids]#取出显示为True的框的xxyy坐标
                final_cls_inds = final_cls_inds[ids]

                all_results.append((final_boxes, final_cls_inds, final_scores))


        return all_results



