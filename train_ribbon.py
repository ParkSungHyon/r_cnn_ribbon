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

# register_coco_instances() : COCO데이터셋(이미지+어노테이션)을 Detectron2에서 사용할 수 있도록 MetadataCatalog에 저장
#                             - 경로 : detectron2.data.datasets.register_coco_instances()
#                                     detectron2.data.datasets.MetadataCatalog
#                             - 메타데이터 카탈로그는 Detectron2에서 사용되는 모든 데이터셋에 대한 정보를 보유하는 중앙 저장소
#                             - register_coco_instances(데이터셋이름:어노테이션+이미지 한쌍을 지칭함, {}, 어노테이션파일(.json파일)경로, 이미지파일 폴더의 경로)
#                             - MetadataCatalog.get(데이터셋이름) : 특정 데이터셋 호출 가능
#                             - Detectron2에서 제공하는 다양한 툴을 사용하기 위해 저장 예: COCO 평가, 다른 COCO 형식의 모델에서 사전 학습된 가중치 사용 등)

# DATASETS.TRAIN = ("어노테이션 폴더경로", "이미지 폴더경로"): 모델 훈련시 사용될 학습 데이터셋을 지정
#                             - 경로 : detectron2.config.get_cfg.DATASETS.TRAIN
#                             - 폴더 내부에는 이미지 파일과 어노테이션 파일이 각각 하나의 파일로 구성되어 있어야함

# --> register_coco_instances()로 데이터셋을 저장, cfg.DATASETS.TRAIN 으로 학습할 데이터 경로를 지정

# 경로 설정
train_json_path = "/Users/b31/Documents/ribbon/train/train_json"
train_img_path = "/Users/b31/Documents/ribbon/train/train_img"
val_json_path = "/Users/b31/Documents/ribbon/val/val_json"
val_img_path = "/Users/b31/Documents/ribbon/val/val_img"

# train_json_path 폴더에서 모든 .json 파일 검색
train_json_files = [f for f in os.listdir(train_json_path) if f.endswith('.json')]
val_json_files = [f for f in os.listdir(val_json_path) if f.endswith('.json')]

# 각각의 JSON 파일마다 데이터셋 등록

for train_json_file in train_json_files:
    train_dataset_name = os.path.splitext(train_json_file)[0]
    register_coco_instances(train_dataset_name, {},os.path.join(train_json_path, train_json_file), train_img_path)

for val_json_file in val_json_files:
    val_dataset_name = os.path.splitext(val_json_file)[0]
    register_coco_instances(val_dataset_name, {}, os.path.join(val_json_path, val_json_file), val_img_path)
    
# 사전 훈련된 모델 불러오기
cfg = get_cfg()
cfg.merge_from_file("/Users/b31/Documents/r_cnn_ribbon2/faster_rcnn_R_50_FPN_3x.yaml")

# 데이터셋 설정
cfg.DATASETS.TRAIN = ("/Users/b31/Documents/ribbon/train/train_json", "/Users/b31/Documents/ribbon/train/train_img",)
cfg.DATASETS.VAL = ("/Users/b31/Documents/ribbon/val/val_json", "/Users/b31/Documents/ribbon/val/val_img",)
cfg.DATASETS.TEST = ()

# 데이터 로더 설정
cfg.DATALOADER.NUM_WORKERS = 2

cfg.OUTPUT_DIR = "/Users/b31/Documents/r_cnn_ribbon2/result"

# 모델 훈련
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# 검증 데이터셋에서 모델 평가
evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = trainer.data_loader.build_test_loader(cfg, val_dataset_name)
inference_on_dataset(trainer.model, val_loader, evaluator)

# 훈련된 모델을 Ribbon.pth로 저장
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
target_model_path = os.path.join(cfg.OUTPUT_DIR, "ribbon_detect.pth")
shutil.move(trained_model_path, target_model_path)