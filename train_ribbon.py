import os
import json
import random
import shutil
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger

setup_logger()

def merge_coco_annotations(json_files):
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    annotation_id = 1

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        if not merged_data["categories"]:
            merged_data["categories"] = data["categories"]

        for image in data["images"]:
            image_id = image["id"]
            merged_data["images"].append(image)

            for annotation in data["annotations"]:
                if annotation["image_id"] == image_id:
                    annotation["id"] = annotation_id
                    annotation_id += 1
                    merged_data["annotations"].append(annotation)

    return merged_data

train_json_dir = "/Users/b31/Documents/ribbon/train/train_json"
train_img_dir = "/Users/b31/Documents/ribbon/train/train_img"
val_json_dir = "/Users/b31/Documents/ribbon/val/val_json"
val_img_dir = "/Users/b31/Documents/ribbon/val/val_img"

train_json_files = []
for filename in os.listdir(train_json_dir):
    if filename.endswith(".json"):
        train_json_files.append(os.path.join(train_json_dir, filename))

val_json_files = []
for filename in os.listdir(val_json_dir):
    if filename.endswith(".json"):
        val_json_files.append(os.path.join(val_json_dir, filename))

# Merge the annotations for the train dataset
merged_train_data = merge_coco_annotations(train_json_files)
merged_train_file = os.path.join(train_json_dir, "merged_train_annotations.json")
with open(merged_train_file, "w") as f:
    json.dump(merged_train_data, f)

# Merge the annotations for the validation dataset
merged_val_data = merge_coco_annotations(val_json_files)
merged_val_file = os.path.join(val_json_dir, "merged_val_annotations.json")
with open(merged_val_file, "w") as f:
    json.dump(merged_val_data, f)

register_coco_instances("ribbon_train", {}, merged_train_file, train_img_dir)
register_coco_instances("ribbon_val", {}, merged_val_file, val_img_dir)

cfg = get_cfg()
cfg.merge_from_file("/Users/b31/Documents/r_cnn_ribbon2/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("/Users/b31/Documents/r_cnn_ribbon2/Base-RCNN-FPN.yaml")

cfg.DATASETS.TRAIN = ("ribbon_train",)
cfg.DATASETS.VAL = ("ribbon_val",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one
lass: ribbon
cfg.OUTPUT_DIR = "/Users/b31/Documents/r_cnn_ribbon2"

# Train the model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the model on the validation dataset
evaluator = COCOEvaluator("ribbon_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = trainer.data_loader.build_test_loader(cfg, "ribbon_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

# Save the trained model as Ribbon.pth
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
target_model_path = os.path.join(cfg.OUTPUT_DIR, "Ribbon.pth")
shutil.move(trained_model_path, target_model_path)