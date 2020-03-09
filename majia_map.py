import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import Resizer, Normalizer, collater, MaJiaDataset, CocoDataset
import cv2
from src.config import COCO_CLASSES, colors
import os


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(
        a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(
        a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


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
    # print('mrec:', mrec)
    mpre = np.concatenate(([0.], precision, [0.]))
    # print('mpre:', mpre)
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # print('mpre envelope:', mpre)
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # print('mrec change value:', i)
    # and sum (\Delta recall) * prec
    # print('mrec[i + 1] - mrec[i]) * mpre[i + 1]:', (mrec[i + 1] - mrec[i]) * mpre[i + 1])
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(
        dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(
                2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # 显示一下
            # if boxes.shape[0] > 0:
            #     print(boxes, scores)
            #     image_info = dataset.coco.loadImgs(dataset.image_ids[index])[0]
            #     path = os.path.join(dataset.root_dir, 'images',  image_info['file_name'])
            #     output_image = cv2.imread(path)
            #     # image_id = dataset.image_ids[index]
            #     # output_image = cv2.imread(str(image_id))
            #
            #     for box_id in range(boxes.shape[0]):
            #         pred_prob = float(scores[box_id])
            #         if pred_prob < score_threshold:
            #             continue
            #         # print(boxes)
            #         pred_label = int(labels[box_id])
            #         xmin, ymin, xmax, ymax = np.int0(boxes[box_id, :])
            #         color = colors[pred_label]
            #         cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            #         text_size = \
            #         cv2.getTextSize(COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            #
            #         cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
            #                       -1)
            #         cv2.putText(
            #             output_image, COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob,
            #             (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            #             (255, 255, 255), 1)
            #     # print("{}/{}_prediction.jpg".format(opt.output, image_id[:-4]))
            #     # cv2.imwrite("{}/{}_prediction.jpg".format(opt.output, image_id[-10:-4]), output_image)
            #     cv2.imshow('1', output_image)
            #     key = cv2.waitKey(0)
            #     if key == ord('q'):
            #         exit()



            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            # print(indices.shape[0])
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(
                    image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))
            # print(all_detections)
            # exit()
            print('{}/{}'.format(index + 1, len(dataset)), end='\r')
    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(
        generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        # annotations = generator._get_annotation(i)
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4]
                                                    == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(
        generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
        # print('\n')
        # print(len(true_positives))
        # print(len(false_positives))
        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        print('[{}]   recall:{}, precision:{}, ap:{}'.format(label,  recall, precision, average_precision))
    print('\nmAP:')
    avg_mAP = []
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        avg_mAP.append(average_precisions[label][0])
    print('avg mAP: {}'.format(np.mean(avg_mAP)))
    return np.mean(avg_mAP), average_precisions


if __name__ == '__main__':
    # 测试码架的map
    model = torch.load('trained_models/signatrix_efficientdet_majia.pth').module
    model.cuda()
    test_set = MaJiaDataset(root_dir="/home/pc/work/data/majia", label_txt="/home/pc/work/data/majia/data_02.txt",
                               transform=transforms.Compose([Normalizer(), Resizer()]))
    evaluate(test_set, model, score_threshold=0.5)

    # 测试coco的map
    # model = torch.load('trained_models/signatrix_efficientdet_coco.pth').module
    # model.cuda()
    # test_set = CocoDataset(root_dir='/disk4t/data/coco/data/coco', set="val2017",
    #                         transform=transforms.Compose([Normalizer(), Resizer()]))
    # evaluate(test_set, model, score_threshold=0.5)