# Stage 3 - Face Verification

# verify_faces.py
# use L2 norm to verify identities within tracks
# if l2 norm > 1.24 -> split track (minimum 100 frames)
import os
from deepface import DeepFace
import numpy as np
from PIL import Image
import time
import cv2
import tensorflow as tf

def l2_similarity(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

def load_image(image_path):
    return Image.open(image_path)

def verify_faces(tracks, all_images, video_folder, threshold=1.24, min_frames=100, batch_size=16):
    pickle_path = os.path.join(video_folder, 'updated_tracks.pkl')
    if os.path.exists(pickle_path):
        return load_tracks(pickle_path)
    updated_tracks = []
    for track_num, track in tracks.items():
        no_img = 0
        prefix = extract_prefix(track[0].url)
        min_frame = extract_frame_number(track[0].url)
        max_frame = extract_frame_number(track[-1].url)
        if len(track) < 0.3 * (max_frame - min_frame): # skip track if 30% or less of clip meets bounding box requirement
            # print(f"TRACK OF LENGTH {max_frame - min_frame} SKIPPED")
            continue
        else: # fill in missing frames of clip
            # print(f"Length - Range | {len(track)} {max_frame - min_frame}")
            filled_track = [all_images[construct_frame_url(prefix, i)] for i in range(min_frame, max_frame + 1)]
            track = filled_track
            tracks[track_num] = filled_track
        features = []
        for i in range(0, len(track), batch_size):
            batch_imgs = track[i:i + batch_size]
            images = []
            for img in batch_imgs:
                if img is None:
                    images.append(None)
                    no_img += 1
                    # print(f"NO IMAGE {no_img}")
                    continue
                image = cv2.imread(img.url)
                face = Image.fromarray(image)
                images.append(np.array(face))
            if not images:
                continue
            try:
                start_time = time.time()
                for image in images:
                    if image is None:
                        features.append(None)
                        continue
                    embedding = DeepFace.represent(
                        img_path=image,
                        model_name='ArcFace',
                        enforce_detection=False,
                        detector_backend='skip'
                    )
                    features.append(embedding)
                end_time = time.time()
                calc_time = end_time - start_time
                # print(f"ArcFace Processing Time: {calc_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing images: {e}")
                continue

        current_id = [track[0]]  # jpgs of same identity
        print("LENGTH", len(features), len(track))
        for i in range(1, len(features)):
            if features[i] is None or features[i - 1] is None:
                if len(current_id) >= min_frames: updated_tracks.append(current_id)
                current_id = []  # new identity track
                continue

            sim = l2_similarity(np.array(features[i - 1][0].get('embedding')), np.array(features[i][0].get('embedding')))

            if sim > threshold:
                # print(f"ARCFACE SPLIT: {track[i - 1].url} - {track[i].url}")
                if len(current_id) >= min_frames: updated_tracks.append(current_id)
                current_id = [track[i]]  # new identity track
            else:
                current_id.append(track[i])
        if len(current_id) > min_frames: updated_tracks.append(current_id)
    save_tracks(updated_tracks, pickle_path)
    return updated_tracks