import os
import argparse
import torch
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer, MaJiaDataset
from src.config import COCO_CLASSES, colors
import cv2
import shutil
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="trained_models1/signatrix_efficientdet_majia.pth")
    parser.add_argument("--output", type=str, default="predictions")
    args = parser.parse_args()
    return args



def test(opt):
    model = torch.load(opt.pretrained_model).module
    model.cuda()
    # dataset = CocoDataset(opt.data_path, set='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset = MaJiaDataset(root_dir="/home/pc/work/data/majia", label_txt="/home/pc/work/data/majia/data_02.txt",
                            transform=transforms.Compose([Normalizer(), Resizer()]))
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    # fw = open('majia_pred.txt', 'w')
    box_count = 0

    for index in range(len(dataset)):
        data = dataset[index]
        scale = data['scale']
        with torch.no_grad():
            scores, labels, boxes, landmarks = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale
            landmarks /= scale

        image_id = dataset.ids[index]
        # fw.write('# {}\n'.format(str(image_id)))
        # fw1 = open('detection-results/{}.txt'.format(os.path.basename(str(image_id))[:-4]), 'w')

        if boxes.shape[0] > 0:
            # image_info = dataset.coco.loadImgs(dataset.image_ids[index])[0]
            # path = os.path.join(dataset.root_dir, 'images', dataset.set_name, image_info['file_name'])
            # output_image = cv2.imread(path)
            image_id = dataset.ids[index]
            output_image = cv2.imread(str(image_id))

            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt.cls_threshold:
                    continue
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = np.int0(boxes[box_id, :].cpu().numpy())
                color = colors[pred_label]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                for k in range(4):
                    cv2.circle(output_image, (landmarks[box_id][2 * k], landmarks[box_id][2 * k + 1]), 2, (0, 255, 0),
                               2)
                # fw.write('{} {} {} {} {} {} {}\n'.format(xmin, ymin, xmax, ymax, pred_prob, 'majia', 1))
                # fw1.write('{} {} {} {} {} {}\n'.format('majia', pred_prob, xmin, ymin, xmax, ymax))
            # print("{}/{}_prediction.jpg".format(opt.output, image_id[:-4]))
            # cv2.imwrite("{}/{}_prediction.jpg".format(opt.output, image_id[-10:-4]), output_image)
            cv2.imshow('1', output_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()
            #     box_count += 1
    # print(box_count)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
