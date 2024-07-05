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

# def verify_faces(tracks, threshold=1.3, min_frames=100):
#     updated_tracks = []
#     for track_num, track in tracks.items():
#         if len(track) < min_frames:
#             print(f"TRACK OF LENGTH {len(track)} SKIPPED")
#             continue
#         track.sort(key=lambda img: img.url)
#         try:
#             start_time = time.time()
#             features = []
#             for img in track:
#                 image = cv2.imread(img.url)
#                 if image is None:
#                     print("NO IMAGE")
#                     continue
#                 cropped_face = image[img.y1:img.y2, img.x1:img.x2]
#                 face = Image.fromarray(cropped_face)
#                 embedding = DeepFace.represent(
#                     img_path=np.array(face), 
#                     model_name='ArcFace', 
#                     enforce_detection=False, 
#                     detector_backend='skip',
#                 )
#                 features.append(embedding)
#             # features = [DeepFace.represent(img_path=img.url, model_name='ArcFace', enforce_detection=False) for img in track]
#             end_time = time.time()
#             calc_time = end_time - start_time
#             print(f"ArcFace Processing Time: {calc_time:2f} seconds")
#         except Exception as e:
#             print(f"Error processing images: {e}")

#         current_id = [track[0]] # jpgs of same identity
#         # print(len(features))
#         for i in range(1, len(features)): # len(features) == len(track)
#             if features[i] is None or features[i-1] is None:
#                 print("features[i] is None or features[i-1] is None")
#                 continue
#             sim = l2_similarity(np.array(features[i-1][0].get('embedding')), np.array(features[i][0].get('embedding')))
#             print(i-1, i, "Similarity:", sim)
#             if sim > threshold:
#                 # print(f"DIFFERENT IDENTITY: {track[i-1].url} - {track[i].url}")
#                 if len(current_id) >= min_frames:
#                     updated_tracks.append(current_id)
#                 current_id = [track[i]] # new identity track
#             else:
#                 current_id.append(track[i])
#     return updated_tracks


def verify_faces(folder_path, threshold=1.3, min_frames=100):
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


    updated_tracks = []
    similarities = []

    # Get list of image files in the directory
    file_entries = [entry for entry in os.scandir(folder_path)]
    file_entries.sort(key=lambda e: e.name)

    if len(file_entries) < min_frames:
        print(f"FOLDER OF LENGTH {len(file_entries)} SKIPPED")
        return updated_tracks, None, None, None, None

    try:
        start_time = time.time()
        features = []
        for file_entry in file_entries:
            image = cv2.imread(file_entry.path)
            if image is None:
                print("NO IMAGE")
                continue
            # Assuming face detection has already been done and crop coordinates are known
            # cropped_face = image[y1:y2, x1:x2]  # Use appropriate coordinates
            cropped_face = image  # Using the whole image if face detection is not available
            face = Image.fromarray(cropped_face)
            embedding = DeepFace.represent(
                img_path=np.array(face), 
                model_name='ArcFace', 
                enforce_detection=False, 
                detector_backend='skip',
            )
            features.append(embedding)
        end_time = time.time()
        calc_time = end_time - start_time
        print(f"ArcFace Processing Time: {calc_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing images: {e}")
        return updated_tracks, None, None, None, None

    current_id = [file_entries[0]] # jpgs of same identity

    for i in range(1, len(features)): # len(features) == len(file_entries)
        if features[i] is None or features[i-1] is None:
            print("features[i] is None or features[i-1] is None")
            continue
        sim = l2_similarity(np.array(features[i-1][0].get('embedding')), np.array(features[i][0].get('embedding')))
        similarities.append(sim)
        print(i-1, i, "Similarity:", sim)
        if sim > threshold:
            if len(current_id) >= min_frames:
                updated_tracks.append(current_id)
            current_id = [file_entries[i]] # new identity track
        else:
            current_id.append(file_entries[i])

    # Calculate statistics
    if similarities:
        min_sim = min(similarities)
        median_sim = median(similarities)
        avg_sim = sum(similarities) / len(similarities)
        max_sim = max(similarities)
    else:
        min_sim = median_sim = avg_sim = max_sim = None

    return updated_tracks, min_sim, median_sim, avg_sim, max_sim

# Example usage
folder_path = "processed-videos/Scarlett Johansson on Being a Movie Star vs. Being an Actor ｜ W Magazine/frames"
updated_tracks, min_sim, median_sim, avg_sim, max_sim = verify_faces(folder_path)

print(f"Updated Tracks: {len(updated_tracks)}")
print(f"Min Similarity: {min_sim}")
print(f"Median Similarity: {median_sim}")
print(f"Average Similarity: {avg_sim}")
print(f"Max Similarity: {max_sim}")