import torch
import numpy as np
import cv2
import os
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
import transforms


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores


def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in tqdm(range(num_cls)[1:]):
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_2d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        # print(recall, precision)
    return all_ap


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


if __name__ == "__main__":
    from my_dataset import VOCDataSet
    from tqdm import tqdm
    import numpy as np

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

    # eval_dataset = VOCDataset(root_dir='C:/Users/86153/Desktop/hw2_v2/voctest/VOCdevkit/VOC2007',
    #                           resize_size=[800, 1333],
    #                           split='test', use_difficult=False, is_train=False, augment=None)
    # print("INFO===>eval dataset has %d imgs" % len(eval_dataset))
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
    #                                           collate_fn=eval_dataset.collate_fn)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()]),
        "test": transforms.Compose([transforms.ToTensor()]),
    }
    train_dataset = VOCDataSet('voctest', "2007", data_transform["test"], "test.txt")
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_sampler=None,
                                                    pin_memory=True,
                                                    num_workers=1,
                                                    collate_fn=train_dataset.collate_fn)

    # model=FCOSDetector(mode="inference")
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load("./checkpoint/model_20.pth",map_location=torch.device('cpu')))
    model.eval()
    print("===>success loading model")

    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []
    num = 0
    for img, targets in tqdm(train_data_loader):
        with torch.no_grad():
            img = torch.unsqueeze(img[0],dim=0)
            predictions = model(img.to(device))[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            pred_boxes.append(predict_boxes)
            pred_classes.append(predict_classes)
            pred_scores.append(predict_scores)
            # out=model(img.cuda())
            # pred_boxes.append(out[2][0].cpu().numpy())
            # pred_classes.append(out[1][0].cpu().numpy())
            # pred_scores.append(out[0][0].cpu().numpy())
        targets = targets[0]
        gt_boxes.append(targets['boxes'].numpy())
        gt_classes.append(targets['labels'].numpy())
        num += 1


    pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
    print('sort over!')
    all_AP = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5,
                        21)
    print('evalap2d over!')
    print("all classes AP=====>\n")
    for key, value in tqdm(all_AP.items()):
        print('ap for {} is {}'.format(key, value))
    mAP = 0.
    for class_id, class_mAP in tqdm(all_AP.items()):
        mAP += float(class_mAP)
    mAP /= (21 - 1)
    print("mAP=====>%.3f\n" % mAP)

