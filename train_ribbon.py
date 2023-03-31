import os
import shutil
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Register the COCO-format dataset
data_dir = "path/to/train/data"
register_coco_instances("ribbon_train", {}, os.path.join(data_dir, "annotations.json"), data_dir)

# Create the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ribbon_train",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Assuming only 1 class (ribbon)

cfg.OUTPUT_DIR = "path/to/save/model"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Rename the trained model to Ribbon.h5
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
target_model_path = os.path.join(cfg.OUTPUT_DIR, "Ribbon.h5")
shutil.move(trained_model_path, target_model_path)
