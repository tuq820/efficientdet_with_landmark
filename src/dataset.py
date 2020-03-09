import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        path = os.path.join(self.root_dir, 'images', image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def label_to_name(self, label):
        return [k for k, v in self.coco_labels.items() if v == label]

    def num_classes(self):
        return 80


class MaJiaDataset(Dataset):
    """MaJia dataset."""

    def __init__(self, root_dir, label_txt, transform=None, shuffle=True, keep_difficult=False):
        """
        Args:
            root_dir (string): MaJia directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.annos, self.ids = self._read_image_ids(label_txt)
        # if shuffle:
        #     random.shuffle(self.ids)
        self.keep_difficult = keep_difficult
        self.load_classes()

    def _read_image_ids(self, image_sets_file):
        ids = []
        annos = {}
        with open(image_sets_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    name = line[1:].strip()
                    name = str(name).replace('/home/a/video/from_ar/', '/home/pc/work/data/majia/')
                    ids.append(name)
                    annos[name] = []
                    continue
                annos[name].append(line)
        return annos, ids

    def load_annotations(self, idx):
        image_id = self.ids[idx]
        objects = self.annos[image_id]
        # print(objects)
        boxes = []
        labels = []
        is_difficult = []
        # annotations = np.zeros((0, 5))
        annotations = np.zeros((0, 5+8))   # 8 is landmarks length
        for item in objects:
            infos = item.strip().split(' ')

            landmarks = infos[5:]
            landmarks = [float(i) for i in landmarks]
            landmark_length = len(landmarks)

            class_name = infos[0]
            # we're only concerned with clases in our list
            # print(self.class_dict)
            if class_name in self.class_dict:
                x1 = float(infos[1])
                y1 = float(infos[2])
                x2 = float(infos[3])
                y2 = float(infos[4])
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult.append(0)

                # annotation = np.zeros((1, 5)) # only detect
                annotation = np.zeros((1, 5+landmark_length))  # include landmark
                annotation[0, :4] = [x1, y1, x2, y2]
                annotation[0, 4] = self.class_dict[class_name]#np.array(self.class_dict[class_name], dtype=np.int64)

                annotation[0, 5:5+landmark_length] = landmarks   # 关键点，不需要就注释 掉

                annotations = np.append(annotations, annotation, axis=0)
        return annotations

    def _read_image(self, image_id):
        image = cv2.imread(str(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.

    def load_classes(self):
        # if the labels file exists, read in the class names
        label_file_name = os.path.join(self.root_dir , "classes.txt")
        # print(label_file_name)
        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            # classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            print("WIDER Labels read from file: " + str(self.class_names))

        else:
            print("No labels file, using default WIDER classes.")
            self.class_names = ('BACKGROUND',
                                'face')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = self._read_image(image_id)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # path = os.path.join(self.root_dir, 'images',
        #                     self.set_name, image_info['file_name'])
        path = os.path.join(self.root_dir, 'images', image_info['file_name'])
        # print('img path:', path)
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def show_image(img, annot):
        font = cv2.FONT_HERSHEY_COMPLEX
        for i in range(annot.shape[0]):
            box = np.int0(annot[i][0:4])
            label = annot[i][4]
            landmark = np.int0(annot[i][5:13])
            cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), (0, 255, 0), 1)
            cv2.putText(img, str(label), tuple(box[0:2]), font, 1, (0, 0, 255), 2)
            for j in range(landmark.shape[0]//2):
                cv2.circle(img, (landmark[2*j], landmark[2*j+1]), 2, (255, 0, 0), 3)
        cv2.imshow('1', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def label_to_name(self, label):
        return [k for k, v in self.class_dict.items() if v == label]

    def num_classes(self):
        return 1



def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5+8)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5+8)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale
        annots[:, 5:13] *= scale  # 关键点，不需要就注释掉

        # MaJiaDataset.show_image(image, annots)

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            # 关键点
            k = annots[:, 5::2].copy()
            annots[:, 5::2] = cols - k

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
