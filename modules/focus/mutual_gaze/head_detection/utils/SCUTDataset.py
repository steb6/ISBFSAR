# Following the tutorial in
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# with the dataset in
# https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release


import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import random
from modules.focus.mutual_gaze.head_detection.utils.misc import get_transform


class SCUTDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.transforms = transforms
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "SCUT_HEAD_Part_A", "JPEGImages"))))
        self.imgs += list(sorted(os.listdir(os.path.join(root, "SCUT_HEAD_Part_B", "JPEGImages"))))
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        if "PartA" in self.imgs[idx]:
            data_path = os.path.join(self.root, "SCUT_HEAD_Part_A")
        else:
            data_path = os.path.join(self.root, "SCUT_HEAD_Part_B")
        # load images and annotation
        img_path = os.path.join(data_path, "JPEGImages", self.imgs[idx])
        annotations_path = os.path.join(data_path, "Annotations", self.imgs[idx]).replace(".jpg", ".xml")
        img = Image.open(img_path).convert("RGB")
        root = ET.parse(annotations_path).getroot()
        # Get bounding boxes
        bboxes = list()
        for elem in root:
            if elem.tag == 'object':
                for e in elem:
                    if e.tag == 'bndbox':
                        xmin, ymin, xmax, ymax = e
                        bboxes.append((int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)))
        if len(bboxes) == 0:
            return self.__getitem__(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((len(bboxes),), dtype=torch.int64)

        # individual image id
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Remove boxes when area = 0
        positive_area = area > 0
        boxes = boxes[positive_area]
        area = area[positive_area]

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img = img / 255.  # Normalize

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import cv2
    import modules.focus.transforms as T

    data = SCUTDataset("D:/SCUT", get_transform(True))

    for i, attr in data:
        bb = attr['boxes']
        i = np.array(i).swapaxes(0, 2).swapaxes(0, 1)
        i = np.ascontiguousarray(i, dtype=np.uint8)
        for bbox in bb:
            xa, ya, xb, yb = bbox[0].int().item(), bbox[1].int().item(), bbox[2].int().item(), bbox[3].int().item()
            i = cv2.rectangle(i, (xa, ya), (xb, yb), (255, 0, 0), 2)
        cv2.imshow("", i)
        cv2.waitKey(0)
