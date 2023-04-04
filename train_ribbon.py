import os
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

# Register the dataset
import os
from detectron2.data.datasets import register_coco_instances

# 경로 설정
train_json_path = "/Users/b31/Documents/ribbon/train/train_json"
train_img_path = "/Users/b31/Documents/ribbon/train/train_img"
val_json_path = "/Users/b31/Documents/ribbon/val/val_json"
val_img_path = "/Users/b31/Documents/ribbon/val/val_img"

# train_json_path 폴더에서 모든 .json 파일 검색
train_json_files = [f for f in os.listdir(train_json_path) if f.endswith('.json')]
val_json_files = [f for f in os.listdir(val_json_path) if f.endswith('.json')]


# 각각의 JSON 파일마다 데이터셋 등록
train_dataset_names = []
for train_json_file in train_json_files:
    train_dataset_name = os.path.splitext(train_json_file)[0]
    register_coco_instances(train_dataset_name, {}, os.path.join(train_json_path, train_json_file + ".json"), train_img_path)
    train_dataset_names.append(train_dataset_name)

val_dataset_names = []
for val_json_file in val_json_files:
    val_dataset_name = os.path.splitext(val_json_file)[0]
    register_coco_instances(val_dataset_name, {}, os.path.join(val_json_path, val_json_file + ".json"), val_img_path)
    val_dataset_names.append(val_dataset_name)

# Load the pre-trained model
cfg = get_cfg()
cfg.merge_from_file("/Users/b31/Documents/r_cnn_ribbon2/faster_rcnn_R_50_FPN_3x.yaml")

# Dataset settings
cfg.DATASETS.TRAIN = tuple(train_dataset_names)
cfg.DATASETS.VAL = tuple(val_dataset_name)
cfg.DATASETS.TEST = ()

print(DatasetCatalog.list())
print(MetadataCatalog.get(train_dataset_name))
print(MetadataCatalog.get(val_dataset_name))

# Dataloader settings
cfg.DATALOADER.NUM_WORKERS = 2

cfg.OUTPUT_DIR = "/Users/b31/Documents/r_cnn_ribbon2/result"

# Train the model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the model on the validation dataset
evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = trainer.data_loader.build_test_loader(cfg, val_dataset_name)
inference_on_dataset(trainer.model, val_loader, evaluator)

# Save the trained model as Ribbon.pth
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
target_model_path = os.path.join(cfg.OUTPUT_DIR, "ribbon_detect.pth")
shutil.move(trained_model_path, target_model_path)