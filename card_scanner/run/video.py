import sys
import os

import cv2

# Dynamically add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from card_scanner.tools import scanner
from card_scanner.tools import tracker as track
from card_scanner.tools import viewer
from card_scanner.tools import detector as detect
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import datetime
import time
import os

weights = "weights/yolov11s_finetune_weights.pt"
weights_quantized = "weights/yolov11s_finetune_weights_int8_openvino_model"
model = YOLO(weights_quantized, task="segment")
video_path = 'test_input_video.mp4'
size = 640
scoreThreshold = .5
use_object_tracking = True
show_video = False
multithreading_enabled = True
multiprocessing_enabled = False # Can use multithreading OR multiprocessing

# detection = {
#     bbox,mask,score,label,track_id,hash, card_image,match
# }

def center_crop(frame, target_size=(640, 640)):
    h, w, _ = frame.shape
    crop_h, crop_w = target_size

    # Calculate cropping coordinates
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)
    end_x = start_x + crop_w
    end_y = start_y + crop_h

    # Perform cropping
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    return cropped_frame

def main():
    # Record the start time
    start_time = time.time()

    print("Starting process - inference with video")

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error: Unable to open video.")
        return

    # Initialize the detector
    detector = detect.Detector(model)

    # Initialize tracker
    tracker = track.Tracker()

    # Keep track of track_ids matched to cards
    tracked_matches = {}

    # Initialize the thread pool
    if multithreading_enabled is True:
        print("MultiTHREADING is ENABED - Creating thread pool")
        pool = ThreadPoolExecutor(max_workers=os.cpu_count())
    elif multiprocessing_enabled is True:
        print("MultiPROCESSING is ENABED - Creating process pool")
        pool = multiprocessing.Pool(12)
    else:
        print("Multithreading and multiprocessing are DISABLED")

    # Initialize the output video
    # output_path = 'output_video.mp4'
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video.get(cv2.CAP_PROP_FPS))

    frame_builder = viewer.VideoFrameBuilder(fps=fps,rows=1,num_images_per_frame=1, is_video=True, max_frame_size=1080)
    # Define the codec and create a VideoWriter object
    # codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    # output_video = cv2.VideoWriter(f'output/{output_path}', codec, fps, (size, size))

    while True:
        # Read the next frame
        ret, image_original = video.read()

        # Check if the frame was read successfully
        if not ret:
            # End of video
            break

        # Resize
        image_original = center_crop(image_original, (size, size))

        image_copy = image_original.copy()

        # Get detected objects
        detections = detector.detect_objects(image_original, scoreThreshold)

        if use_object_tracking is True:
            # Add object tracking IDs to detections
            tracker.track_objects(detections)

        # Reinstate matches for already tracked objects
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id in tracked_matches:
                detection['match'] = tracked_matches[track_id]

        if multithreading_enabled is True:
            # Make hashes and matches to detections threaded
            scanner.get_matches_threaded(pool, image_original, detections, mirror=False)
        elif multiprocessing_enabled is True:
            scanner.get_matches_multiprocessed(pool, image_original, detections, mirror=False)
        else:
            # Make hashes and matches to detections non-threaded
            scanner.process_masks_to_cards(image_original, detections, mirror=False)
            scanner.hash_cards(detections)
            scanner.match_hashes(detections)

        # Store tracked matches
        for detection in detections:
            if 'track_id' in detection and 'match' in detection:
                track_id = detection['track_id']
                match = detection['match']
                if track_id not in tracked_matches:
                    tracked_matches[track_id] = match
                    # print(f'Match found: id {track_id} {match}')


        # Draw elements
        scanner.draw_boxes(image_copy, detections)
        scanner.draw_masks(image_copy, detections)
        scanner.write_card_labels(image_copy, detections)
        scanner.write_track_id(image_copy, detections)

        frame_builder.add_image(image_copy)

        # Write the processed frame to the output video
        #output_video.write(image_copy)

        # if len(detections) > 0:
        #     if 'card_image' in detections[0]:
        #         cv2.imshow('Image', detections[0]['card_image'])
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

        if show_video is True:
            # Display the resized frame
            cv2.imshow('Result', image_copy)

            # Check for key press to exit the loop or save a screenshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Key pressed. Ending early.')
                break
            elif key == ord('s'):  # Check if 'S' key is pressed
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # Write the image to a file
                screenshot_filename = f'output/screenshots/screenshot_{current_time}.png'
                cv2.imwrite(screenshot_filename, image_copy)
                print(f"Screenshot saved to '{screenshot_filename}'")

        frame_builder.create_frame()


    # Release the video capture and writer objects
    video.release()
    # output_video.release()
    frame_builder.release()
    print('Video saved')

    cv2.destroyAllWindows()

    # Record the end time
    end_time = time.time()

    # Print the total time taken
    print("Total time taken: ", round(end_time - start_time, 1), ' seconds')


if __name__ == '__main__':
    main()
