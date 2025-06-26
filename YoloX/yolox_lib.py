import os
import sys
import time
import torch
import cv2
import configparser
from pathlib import Path

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from loguru import logger


COCO_CLASSES_RE = (
    'top',
    'bottom',
    'left',
    'right',
)

COCO_CLASSES = (
    'Defects',
)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names,
        trt_file=None,
        decoder=None,
        device="gpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        # dt = time.time()
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        # print('step2 >>>', time.time()-dt)
        # dt = time.time()
        test_time =time.time()
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()
        #print(time.time() - test_time)
        # print('step3 >>>', time.time()-dt)
        # dt = time.time()
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
            )
        # print('step4 >>>', time.time()-dt)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.1):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

def predictor(model_type):
    config_path = 'info.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    # print(COCO_CLASSES)
    if model_type == 'detect':
        exp_name = 'yolox_m_new.py'
        exp_type = 'yolox-m'
        AI_MODEL = config['INFO']['AI_MODEL']
        coco = COCO_CLASSES

    elif model_type == 'reverse':
        exp_name = 'yolox_m.py'
        exp_type = 'yolox-m'
        AI_MODEL = 'model/model_trt_reverse_220602.pth'
        # AI_MODEL = 'model/0607_revers_model_trt4.pth'
        coco = COCO_CLASSES_RE

    exp = get_exp('exps/example/custom/'+exp_name, exp_type)

    exp.test_conf = 0.1
    exp.nmsthre = 0.7
    exp.test_size = (640, 640)

    net = exp.get_model()
    net.cuda()    # Use GPU
    net.eval()

    trt = True  # Using TensorRT to inference
    fuse = False
    decoder = None   # Using TensorRT to inference

    # print(coco)

    if trt:
        # TensorRT 엔진 사용시만 아래 코드 활성화
        assert not fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(AI_MODEL)
        try:
            if not os.path.exists(trt_file):
                raise FileNotFoundError(f"{trt_file} not found")
            net.head.decode_in_inference = False
            decoder = net.head.decode_outputs
            logger.info("Using TensorRT to inference")
            predictor = Predictor(net, exp, coco, trt_file, decoder, 'gpu', False)
        except Exception as e:
            print("TensorRT model is not found!\n Run python3 tools/trt.py first!")
            print('Error:', e)
            predictor = None
    else:
        # PyTorch로 추론할 때 여기서 Predictor 객체 생성
        trt_file = None
        decoder = None
        predictor = Predictor(net, exp, coco, trt_file, decoder, 'gpu', False)

    return predictor

def yx_detect(predictor, frame):
    # dt = time.time()
    outputs, img_info = predictor.inference(frame)
    # print('step1 >>>', time.time()-dt)
    # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
    #cv2.imwrite('result_frame/'+str(time.time()) + '_result_frame.bmp', frame)
    ratio = img_info["ratio"]
    output = outputs[0]
    if output is not None:
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls_name = output[:, 6]
        scores = output[:, 4] * output[:, 5]
    else:
        bboxes, cls_name, scores = None, None, None
    return frame, bboxes, cls_name, scores
    # return bboxes, cls_name, scores





#if __name__ == '__main__':
    #predictor()
