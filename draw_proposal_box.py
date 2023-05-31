import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_pbox


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights
    model_name = 'resNetFpn-model-11.pth'
    weights_path = "./save_weights/{}".format(model_name)
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    # original_img = Image.open("./test.jpg")
    original_img = Image.open("C:/Users/86153/Desktop/hw2_v2/VOCdevkit/VOC2007/JPEGImages/000020.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)

        # plt.imshow(plot_img)
        # plt.show()
        # # 保存预测的图片结果
        # plot_img.save("test_result.jpg")


if __name__ == '__main__':
    # hyper para
    model_name = 'resNetFpn-model-20.pth'
    original_img = Image.open("C:/Users/86153/Desktop/hw2_v2/voctest/VOCdevkit/VOC2007/JPEGImages/000001.jpg")
    original_img = Image.open("C:/Users/86153/Desktop/hw2_v2/voctest/VOCdevkit/VOC2007/JPEGImages/000025.jpg")
    original_img = Image.open("C:/Users/86153/Desktop/hw2_v2/voctest/VOCdevkit/VOC2007/JPEGImages/000369.jpg")
    original_img = Image.open("C:/Users/86153/Desktop/hw2_v2/voctest/VOCdevkit/VOC2007/JPEGImages/001183.jpg")
    # original_img = Image.open("D:/历史照片/这两年/psc.jfif")  # 0 364 857 2134 # 864 486 1600 2134
    # original_img = Image.open("D:/手机存档/照片/收藏/球球.jpg")  # 0 143 1665 1158
    # original_img = Image.open("test/car_behind_bike.jpg")  # 141 247 1426 1032 # 476 345 801 644

    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights

    weights_path = "./save_weights/{}".format(model_name)
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    # original_img = Image.open("./test.jpg")
    # original_img = Image.open("C:/Users/86153/Desktop/hw2_v2/VOCdevkit/VOC2007/JPEGImages/000019.jpg")
    # original_img = Image.open("test/R-C.png")
    # original_img = Image.open("test/3.jpg")


    original_img = original_img.convert('RGB') # 即使是png图片四通道也转成普通三通道

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    print(img.shape)
    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_pbox(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0,
                             line_thickness=4,
                             font='arial.ttf',
                             font_size=65)
        plt.imshow(plot_img)
        plt.show()



        # plt.imshow(plot_img2)
        # plt.show()


        # 保存预测的图片结果
        # plot_img.save("test_result.jpg")

        def count_Iou(l1, l2):
            x1,y1,x2,y2 = l1
            xg1,yg1,xg2,yg2 = l2
            s1 = (x2 - x1) * (y2 - y1)
            s2 = (xg1 - xg2) * (yg1 - yg2)
            if xg1 > x2 or x1 > xg2 or yg1 > y2 or y1 > yg2:
                return 0
            xi = min(x2,xg2) - max(x1, xg1)
            yi = min(y2,yg2) - max(y1, yg1)
            s_intersec = xi * yi
            return s_intersec/(s1 + s2 - s_intersec)