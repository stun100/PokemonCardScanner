import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import scanner
from card_scanner.tools import scanner
from card_scanner.tools import detector as detect
from card_scanner.tools import tracker as track
from card_scanner.tools import viewer
from ultralytics import YOLO
import cv2
import datetime

weights = "weights/yolov11s_finetune_weights.pt"
weights_quantized = "weights/yolov11s_finetune_weights_int8_openvino_model"
model = YOLO(weights, task="segment")
size = 640
scoreThreshold = .5
save_video = True

# detection = {
#     bbox,mask,score,label,track_id,hash, card_image,match
# }

def main():
    print("Starting process - inference with webcam")
    # Open the webcam
    camera = cv2.VideoCapture(0)

    # Set the desired frames per second (fps)
    # camera.set(cv2.CAP_PROP_FPS, 60)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(camera.get(cv2.CAP_PROP_FPS))

    # Print the current config
    print("Webcam started with resolution:", width, "x", height, 'fps:', fps)

    # Initialize the detector
    Detector = detect.Detector(model)

    # Initialize tracker
    Tracker = track.Tracker()

    # Keep track of track_ids matched to cards
    tracked_matches = {}

    frame_builder = viewer.VideoFrameBuilder(fps=fps, rows=2, num_images_per_frame=6, is_video=True, max_frame_size=1080*2)

    while True:
        image_original = scanner.read_frame(camera, size)
        image_original = cv2.flip(image_original, 1)
        if image_original is None:
            break
        image_copy = image_original.copy()

        frame_builder.add_image(image_copy, 0, "Original Frame")

        # Get detected objects
        detections = Detector.detect_objects(image_original, scoreThreshold)

        # Add object tracking IDs to detections
        Tracker.track_objects(detections)

        # Reinstate matches for already tracked objects
        # for detection in detections:
        #     track_id = detection.get('track_id')
        #     if track_id in tracked_matches:
        #         detection['match'] = tracked_matches[track_id]

        # Make hashes and matches to detections
        scanner.process_masks_to_cards(image_original, detections, mirror=True)
        scanner.hash_cards(detections)
        scanner.match_hashes(detections)

        # Store tracked matches
        for detection in detections:
            if 'track_id' in detection and 'match' in detection:
                track_id = detection['track_id']
                match = detection['match']
                if track_id not in tracked_matches:
                    tracked_matches[track_id] = match
                    print(f'Match found: id {track_id} {match}')

        # Draw elements
        scanner.draw_boxes(image_copy, detections)
        scanner.draw_masks(image_copy, detections)

        scanner.write_track_id(image_copy, detections)

        frame_builder.add_image(image_copy, 1, "ML Detections")

        scanner.write_card_labels(image_original, detections)

        if len(detections) > 0:
            if 'card_image' in detections[0]:
                frame_builder.add_image(detections[0]['card_image'], 3, "Perspective Transform")

        frame_builder.add_image(None, 4, "Hash & Compare")
        frame_builder.add_image(image_original, 5, "Result")
        frame_builder.create_frame()

    frame_builder.release()
    print('Video saved')


if __name__ == '__main__':
    main()

# download all images for all current sets
# hash all to text file
# add all hashes to vp tree
