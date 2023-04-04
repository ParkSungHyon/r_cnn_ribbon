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
train_json_path = "C:/python123/r_cnn_ribbon/ribbon/train/train_json"
train_img_path = "C:/python123/r_cnn_ribbon/ribbon/train/train_img"
val_json_path = "C:/python123/r_cnn_ribbon/ribbon/val/val_json"
val_img_path = "C:/python123/r_cnn_ribbon/ribbon/val/val_img"

# train_json_path 폴더에서 모든 .json 파일 검색
train_json_files = [f for f in os.listdir(train_json_path) if f.endswith('.json')]
val_json_files = [f for f in os.listdir(val_json_path) if f.endswith('.json')]


# 각각의 JSON 파일마다 데이터셋 등록
for train_json_file in train_json_files:
        train_dataset_name = os.path.splitext(train_json_file)[0]
        register_coco_instances(train_dataset_name, {}, os.path.join(train_json_path, train_json_file + ".json"), train_img_path)

for val_json_file in val_json_files:
    val_dataset_name = os.path.splitext(val_json_file)[0]
    register_coco_instances(val_dataset_name, {}, os.path.join(val_json_path, val_json_file + ".json"), val_img_path)

# Load the pre-trained model
cfg = get_cfg()
cfg.merge_from_file("C:/python123/r_cnn_ribbon/faster_rcnn_R_50_FPN_3x.yaml")

# Dataset settings
cfg.DATASETS.TRAIN = (train_dataset_name,)
cfg.DATASETS.VAL = (val_dataset_name,)
cfg.DATASETS.TEST = ()

# Dataloader settings
cfg.DATALOADER.NUM_WORKERS = 2

# Trainer settings
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: ribbon
cfg.OUTPUT_DIR = "C:/python123/r_cnn_ribbon2"

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