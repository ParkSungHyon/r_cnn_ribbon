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
register_coco_instances("ribbon_train", {}, "C:/python123/r_cnn_ribbon/ribbon/train/train_json", "C:/python123/r_cnn_ribbon/ribbon/train/train_img")
register_coco_instances("ribbon_val", {}, "C:/python123/r_cnn_ribbon/ribbon/val/val_json", "C:/python123/r_cnn_ribbon/ribbon/val/val_img")

# Load the pre-trained model
cfg = get_cfg()
cfg.merge_from_file("C:/python123/r_cnn_ribbon/faster_rcnn_R_50_FPN_3x.yaml")

# Dataset settings
cfg.DATASETS.TRAIN = ("ribbon_train",)
cfg.DATASETS.VAL = ("ribbon_val",)
cfg.DATASETS.TEST = ()

# Dataloader settings
cfg.DATALOADER.NUM_WORKERS = 2

# Trainer settings
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: ribbon
cfg.OUTPUT_DIR = "C:/python123/r_cnn_ribbon"

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
