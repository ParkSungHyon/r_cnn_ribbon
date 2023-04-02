from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor


# Load the test dataset metadata
test_metadata = MetadataCatalog.get("ribbon_test")

# Create the test data loader
test_loader = build_detection_test_loader(cfg, "ribbon_test")

# Load the trained model
predictor = DefaultPredictor(cfg)
predictor.model.load_state_dict(torch.load("path/to/Ribbon.pth"))

# Test the model on the validation dataset
for i, batch in enumerate(test_loader):
    # Perform inference
    with torch.no_grad():
        outputs = predictor(batch)

    # Visualize the results
    v = Visualizer(
        batch["image"].numpy()[0],
        metadata=test_metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Results", v.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
