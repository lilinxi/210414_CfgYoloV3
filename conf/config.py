import os

import torch

# -----------------------------------------------------------------------------------------------------------#
# PennFudanConfig
# VocConfig
# DefaultCocoConfig
# -----------------------------------------------------------------------------------------------------------#

CocoNamesPath = os.path.join(os.getcwd(), "conf", "coco.names")
VocNamesPath = os.path.join(os.getcwd(), "conf", "voc.names")

DarkNet53WeightPath = os.path.join(os.getcwd(), "weights", "demo_darknet53_weights.pth")

PennFudanConfig: dict = {
    # 1. 默认配置
    "anchors": [  # 锚框，width * height
        [
            [116, 90], [156, 198], [373, 326]  # 大
        ], [
            [30, 61], [62, 45], [59, 119]  # 中
        ], [
            [10, 13], [16, 30], [33, 23]  # 小
        ]
    ],
    "image_height": 416,  # 输入图片高度
    "image_width": 416,  # 输入图片宽度
    "conf_threshold": 0.5,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": False and torch.cuda.is_available(),  # 是否使用 GPU
    # 2. 数据集专属配置
    "dataset_root": "/Users/limengfan/Dataset/PennFudanPed",
    "weights_path": "/Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs/"
                    "Pennfudan_Test1_Epoch88-Train_Loss5.0447-Val_Loss2.9787.pth",  # 模型权重
    "classes": 1,  # 分类数目
    "labels": [  # 类别标签
        "person",
    ]
}

VocConfig: dict = {
    # 1. 默认配置
    "anchors": [  # 锚框，width * height
        [
            [116, 90], [156, 198], [373, 326]  # 大
        ], [
            [30, 61], [62, 45], [59, 119]  # 中
        ], [
            [10, 13], [16, 30], [33, 23]  # 小
        ]
    ],
    "image_height": 416,  # 输入图片高度
    "image_width": 416,  # 输入图片宽度
    "conf_threshold": 0.05,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": True and torch.cuda.is_available(),  # 是否使用 GPU
    # 2. 数据集专属配置
    # "dataset_root": "/Users/limengfan/Dataset/VOC/VOC2012Train",
    "weights_path": "/Users/limengfan/PycharmProjects/210414_CfgYoloV3/logs/"
                    "Voc_Test1_Epoch223-Train_Loss0.0166-Val_Loss19.6223.pth",  # 模型权重
    "pretrained_weights_path": os.path.join(os.getcwd(), "outer_weights", "demo_darknet53_weights.pth"),  # 预训练模型权重
    "dataset_root": "/home/lenovo/data/lmf/Dataset/voc/VOCtrainval_11-May-2012",
    # "weights_path": "/home/lenovo/data/lmf/210414_CfgYoloV3/logs/"
    #                 "Voc_Test1_Epoch223-Train_Loss0.0166-Val_Loss19.6223.pth",  # 模型权重
    "classes": 20,  # 分类数目
    "labels": [
        line.strip() for line in
        open(VocNamesPath).readlines()
    ] if os.path.exists(VocNamesPath)
    else [print("warn in loading voc.names", VocNamesPath)],
}

DefaultCocoConfig: dict = {
    # 1. 默认配置
    "anchors": [  # 锚框，width * height
        [
            [116, 90], [156, 198], [373, 326]  # 大
        ], [
            [30, 61], [62, 45], [59, 119]  # 中
        ], [
            [10, 13], [16, 30], [33, 23]  # 小
        ]
    ],
    "image_height": 416,  # 输入图片高度
    "image_width": 416,  # 输入图片宽度
    "conf_threshold": 0.5,  # 正确预测框的最小置信度
    "nms_iou_threshold": 0.3,  # 判断预测框重合的最大 iou 阈值
    "cuda": False and torch.cuda.is_available(),  # 是否使用 GPU
    # 2. 数据集专属配置
    "weights_path": "/Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_yolov3_weights.pth",  # 模型权重
    "classes": 80,  # 分类数目
    "labels": [
        line.strip() for line in
        open(CocoNamesPath).readlines()
    ] if os.path.exists(CocoNamesPath)
    else [print("warn in loading coco.names", CocoNamesPath)],
}
