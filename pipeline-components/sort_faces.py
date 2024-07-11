# Stage 2 - Face Tracking
# RetinaFace + SORT
import cv2
from retinaface import RetinaFace
import os
import numpy as np
from sort import Sort
import time
from collections import defaultdict

class Img:
    def __init__(self, url, x1, x2, y1, y2):
        self.url = url
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


def sort_faces(video_folder, batch_size=16, scale_factor=0.15):  # video_folder = processed-videos/video_name
    all_imgs = {}
    detector = RetinaFace(gpu_id=0)
    pickle_path = os.path.join(video_folder, 'tracks.pkl')
    images_path = os.path.join(video_folder, "images.pkl")
    input_folder = os.path.join(video_folder, 'frames')
    if os.path.exists(pickle_path) and os.path.exists(images_path):
        return load_tracks(pickle_path), load_tracks(images_path)
    tracks = defaultdict(list)
    tracker = Sort()  # Initialize the SORT tracker
    img_files = [entry.path for entry in sorted(os.scandir(input_folder), key=lambda e: e.name) if entry.is_file() and entry.name.lower().endswith('.jpg')]
    
    for i in range(0, len(img_files), batch_size):
        start_time = time.time()
        batch_files = img_files[i:i + batch_size]
        images = [cv2.imread(file) for file in batch_files]

        # Scale down the images
        scaled_images = [cv2.resize(image, None, fx=scale_factor, fy=scale_factor) for image in images]
        batch_detections = detector(scaled_images, cv=True)

        for file_path, detections, original_image in zip(batch_files, batch_detections, images):
            filename = os.path.basename(file_path)
            if len(detections) != 1:  # Only consider frames with one detected face
                all_imgs[file_path] = None
                continue
            valid = False
            frame_detections = []  # Faces in the current frame
            # Process detections and prepare for tracking
            box, landmarks, score = detections[0]
            x1, y1, x2, y2 = box

            # Scale the coordinates back up to the original image size
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)

            width = x2 - x1
            height = y2 - y1
            img = Img(file_path, x1, x2, y1, y2)
            all_imgs[file_path] = img
            if width >= 500 and height >= 500:
                valid = True
                frame_detections.append([x1, y1, x2, y2, score])
            if valid:
                np_detections = np.array(frame_detections)
                tracked_objects = tracker.update(np_detections)
                for track in tracked_objects:  # len(tracked_objects) = 0 or 1
                    x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                    tracks[track_id].append(img)
        end_time = time.time()
        detection_time = end_time - start_time
        # print(f"Batch {i // batch_size + 1}: {detection_time:.2f} seconds")
    save_tracks(tracks, pickle_path)
    save_tracks(all_imgs, images_path)
    return tracks, all_imgs
