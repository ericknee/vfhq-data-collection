import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pickle
import re
import os

from scenedetect import detect, ContentDetector
scenedetect -i "videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle.mp4" list-scenes save-images
scene_list = detect('videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle.mp4', ContentDetector())
print(scene_list)

class Img:
    def __init__(self, url, x1, x2, y1, y2):
        self.url = url
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

def extract_frame_number(frame_url):
    # Extract the frame number from the filename using regex
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        frame_number = int(match.group(1))
        prefix = os.path.dirname(frame_url)
        return prefix, frame_number
    return None, None

def construct_frame_url(prefix, frame_number):
    return os.path.join(prefix, f"frame_{frame_number:05d}.jpg")

def load_tracks(file_path):
    with open(file_path, 'rb') as f:
        tracks = pickle.load(f)
    print(f"Tracks loaded from {file_path}")
    return tracks

def edge_difference(image1, image2):
    edges1 = cv2.Canny(image1, 50, 150)
    edges2 = cv2.Canny(image2, 50, 150)
    return np.sum(np.abs(edges1 - edges2)) / (image1.shape[0] * image1.shape[1])

def detect_edge_cuts(frames, threshold=0.1):
    cuts = []
    for i in range(1, len(frames)):
        diff = edge_difference(frames[i-1], frames[i])
        if diff > threshold:
            cuts.append(i)
    return cuts

def ssim_compare(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def detect_ssim_cuts(frames, threshold=0.5):
    cuts = []
    for i in range(1, len(frames)):
        score = ssim_compare(frames[i-1], frames[i])
        if score < threshold:
            cuts.append(i)
    return cuts

def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def detect_optical_flow_cuts(frames, threshold=2.0):
    cuts = []
    for i in range(1, len(frames)):
        flow = calculate_optical_flow(frames[i-1], frames[i])
        if flow > threshold:
            cuts.append(i)
    return cuts

# Example usage
# edge_diffs = []
# tracks = load_tracks('processed-videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle/updated_tracks.pkl')
# for track in tracks:
#     for i in range(1, len(track)):
#         image1 = track[i - 1].url
#         image2 = track[i].url
#         if image1 is None or image2 is None: continue
        
#         img1 = cv2.imread(image1)
#         img2 = cv2.imread(image2)
#         if img1 is None or img2 is None: continue

#         pre1, frame1 = extract_frame_number(image1)
#         pre2, frame2 = extract_frame_number(image2)
        
#         diff = calculate_optical_flow(img1, img2)
#         # edge_diffs.append(diff)
        
#         print(f"{diff}, {frame1}-{frame2}")

# # Calculate statistics
# mean_diff = np.mean(edge_diffs)
# median_diff = np.median(edge_diffs)
# min_diff = np.min(edge_diffs)
# max_diff = np.max(edge_diffs)

# print(f"Mean edge difference: {mean_diff}")
# print(f"Median edge difference: {median_diff}")
# print(f"Min edge difference: {min_diff}")
# print(f"Max edge difference: {max_diff}")
# pre = "processed-videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle/frames/"
# frames = [292, 313, 335, 368, 396, 469, 484, 508]
# for frame in frames:
#     frame2 = frame + 1
#     url1 = construct_frame_url(pre, frame)
#     url2 = construct_frame_url(pre, frame2)
#     img1 = cv2.imread(url1)
#     img2 = cv2.imread(url2)
#     diff = edge_difference(img1, img2)
#     print(f"EDGE {diff}, {frame}-{frame2}")