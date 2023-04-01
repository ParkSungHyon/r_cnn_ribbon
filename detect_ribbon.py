# detect_ribbon.py

import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Load the trained R-CNN model
cfg = get_cfg()
cfg.merge_from_file("Ribbon_R-CNN_config.yaml")  # Set the path to your R-CNN config file
cfg.MODEL.WEIGHTS = "Ribbon.h5"  # Set the path to your trained R-CNN model
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

def find_ribbon_on_screen():
    screenshot = ImageGrab.grab()
    screenshot_np = np.array(screenshot)

    # Resize the input image to handle different ribbon sizes
    resized_image = cv2.resize(screenshot_np, (640, 480))
    
    outputs = predictor(resized_image)
    instances = outputs["instances"].to("cpu")
    ribbons = instances[instances.pred_classes == 0]  # Assuming class_id 0 is for ribbon

    if len(ribbons) > 0:
        bbox = ribbons.pred_boxes.tensor[0]
        return bbox

    return None

while True:
    ribbon = find_ribbon_on_screen()
    if ribbon is not None:
        x1, y1, x2, y2 = ribbon
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Convert the coordinates back to the original screen resolution
        screen_width, screen_height = pyautogui.size()
        center_x = int(center_x * screen_width / 640)
        center_y = int(center_y * screen_height / 480)

        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()
        break

    pyautogui.scroll(-1)  # Scroll down
    time.sleep(1)  # Adjust the sleep time as needed
        # git Add test