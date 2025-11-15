from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, json
from collections import defaultdict
import torch


class GTACarDataset(Dataset):

    def __init__(self, root_dir, split, processor):
        """
        root_dir: ./hw3_dataset
        split: 'train' or 'val'
        annotations.json: ./hw3_dataset/{split}/annotations.json
        images:          ./hw3_dataset/{split}/images/xxx.png
        """
        self.root_dir = root_dir
        self.split = split
        self.processor = processor

        ann_path = os.path.join(root_dir, split, "annotations.json")
        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        anns = coco["annotations"]

        # 把 annotation 依 image_id 分組
        self.anns_per_image = defaultdict(list)
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            self.anns_per_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.root_dir, self.split, "images", file_name)
        image = Image.open(img_path).convert("RGB")

        anns = self.anns_per_image[img_id]

        # HF RT-DETR / DETR 期望 COCO style 的 dict
        target = {
            "image_id":
            img_id,
            "annotations": [
                {
                    "bbox": ann["bbox"],  # [x_min, y_min, w, h]
                    "category_id": ann["category_id"]  # 車 = 0
                } for ann in anns
            ],
        }

        encoded = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        # encoded:
        # pixel_values: (1, 3, H, W)
        # pixel_mask:   (1, H, W)
        # labels:       tuple(len=1)[ dict(boxes, class_labels, area, ...) ]
        pixel_values = encoded["pixel_values"].squeeze(0)
        pixel_mask = encoded["pixel_mask"].squeeze(0)
        labels = encoded["labels"][0]

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]  # list of dicts，長度隨 batch_size

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
    }
