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


def sort_faces(input_folder): # video folder = processed-videos/video
    boxCount = 0
    tracks =  defaultdict(list)
    tracker = Sort()  # Initialize the SORT tracker
    for file_entry in sorted(os.scandir(input_folder), key=lambda e: e.name):
        if file_entry.is_file() and file_entry.name.lower().endswith('.jpg'):
            filename = file_entry.name
            file_path = file_entry.path
            img = cv2.imread(file_path)
            start_time = time.time()
            detections = RetinaFace.detect_faces(file_path) 
            end_time = time.time()
            detection_time = end_time - start_time
            print(f"Detection time for {filename}: {detection_time:.2f} seconds")

            if detections is None:
                continue

            valid = False
            frame_detections = []  # Faces in the current frame
            # Process detections and prepare for tracking
            for key, value in detections.items():
                facial_area = value['facial_area']
                width = facial_area[2] - facial_area[0]
                height = facial_area[3] - facial_area[1]
                score = value['score']
                # Filter out small faces
                if width >= 500 and height >= 500:
                    boxCount += 1
                    print(boxCount)
                    valid = True
                    # SORT data format: [x1, y1, x2, y2, score]
                    frame_detections.append([facial_area[0], facial_area[1], facial_area[2], facial_area[3], score])

            if valid:
                np_detections = np.array(frame_detections)
                tracked_objects = tracker.update(np_detections)

                for track in tracked_objects:
                    x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                    img = Img(file_path, x1, x2, y1, y2)
                    tracks[track_id].append(img)
    return tracks


input_base_folder = 'frames'
sort_faces(input_base_folder)
