import sys
import functools
import os
import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", message="Failed to add <class 'mmengine.visualization.vis_backend.LocalVisBackend'>.*")


def disable_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to suppress print output
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            result = func(*args, **kwargs)
        # Restore stdout
        sys.stdout = sys.__stdout__
        return result
    return wrapper

@disable_print
class Detector:
    def __init__(self, model):
        self.model = model
        
    @disable_print
    def detect_objects(self, img, scoreThreshold):
        # Perform inference
        data = self.model(img)

        detections = []
          # Convert masks to NumPy array
        if data[0].masks:
            masks = data[0].masks.data.cpu().numpy()
            bboxes = data[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
            confidences = data[0].boxes.conf.cpu().numpy()  # Confidence scores
            classes = data[0].boxes.cls.cpu().numpy()  # Class IDs

            # detections = []

            for i in range(len(confidences)):
                if confidences[i] > scoreThreshold:
                    detection = {
                        'bbox': list(map(int, bboxes[i])), #convert to int
                        'score': confidences[i],
                        'mask': masks[i]
                    }
                    detections.append(detection)

        return detections
