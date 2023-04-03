import os
import json
from PIL import Image

def convert_to_coco(json_path, img_dir, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    images = []
    annotations = []
    categories = [{"id": 1, "name": "ribbon"}]  # Assuming ribbon is the only category

    for i, (filename, img_data) in enumerate(data.items()):
        img_path = os.path.join(img_dir, img_data["filename"])
        width, height = Image.open(img_path).size

        images.append({
            "id": i + 1,
            "file_name": img_data["filename"],
            "height": height,
            "width": width
        })

        for region in img_data["regions"]:
            shape = region["shape_attributes"]
            region_attrs = region["region_attributes"]

            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
            bbox = [x, y, w, h]
            area = w * h
            category_id = categories[0]["id"]  # Assuming ribbon is the only category

            annotations.append({
                "id": len(annotations) + 1,
                "image_id": i + 1,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

    output_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f)

    print(f"Converted {len(images)} images to COCO-style annotation file: {output_path}")
