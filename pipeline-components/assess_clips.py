# Stage 4 - Selecting High Quality Clips
# hyperIQA to evaluate image quality
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import shutil
import models
import face_alignment
import cv2
import time


import os

def get_relative_file_paths(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            relative_path = os.path.join(root, file)
            full_path = os.path.join(directory_path, relative_path)
            file_paths.append(full_path)
    return file_paths


def load_img(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_landmarks(fa, img, image):
    bbox = np.array([img.x1, img.y1, img.x2, img.y2])
    landmarks = fa.get_landmarks_from_image(image, detected_faces=[bbox]) # detected_faces = list of np.arrays
    if landmarks is not None:
        return landmarks[0]  # landmarks for first detected face
    return None

def get_all_landmarks(fa, images, batch_size=8):
    landmark_points = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        loaded_images = [cv2.imread(img.url) for img in batch_images]
        for img, image in zip(batch_images, loaded_images):
            landmarks = get_landmarks(fa, img, image) # img = object, image = loaded image
            if landmarks is not None:
                landmark_points.extend(landmarks)
    return landmark_points
            
def calculate_motion(landmarks):
    N = len(landmarks) # N = number of frames in clip
    if N < 2: return 0
    total = 0
    for i in range(N - 1):
        curr = landmarks[i]
        next = landmarks[i + 1]
        distance = np.linalg.norm(next - curr) ** 2
        total += distance
    total /= (N * 98)
    return 0.25 * total + 42.5

            
def calculate_motion(landmarks):
    N = len(landmarks) # N = number of frames in clip
    print(f"\nNUMBER OF LANDMARKS {N}")
    if N < 2: return 0
    total = 0
    for i in range(N - 1):
        curr = landmarks[i]
        next = landmarks[i + 1]
        distance = np.linalg.norm(next - curr) ** 2
        total += distance
    total /= (N * 98)
    return 0.25 * total + 42.5

def assess_clips(tracks, video_folder, frame_threshold=42, clip_threshold=45, alpha=0.5, beta=0.2, hyper_batch_size=8):
    pickle_path = os.path.join(video_folder, 'urls.pkl')
    if os.path.exists(pickle_path):
        return load_tracks(pickle_path)
    final_clips = []
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu', flip_input=False)
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_hyper = model_hyper.to(device)
    model_hyper.train(False)
    # load pre-trained model on koniq-10k dataset
    model_hyper.load_state_dict((torch.load('koniq_pretrained.pkl', map_location=device)))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])
    track_score = 0
    track_buffer = 0
    track_items = 0
    for track in tracks:
        frame_scores = [] # frames for each track
        for i in range(0, len(track), hyper_batch_size):
            images = []
            for j in range(i, i + hyper_batch_size):
                if j >= len(track): break
                if track[j] is None:
                    images.append(None)
                else:
                    images.append(load_img(track[j].url))
            for image in images: # img = Img object
                if image is None:
                    frame_scores.append(-1.0)
                    continue
                image = transforms(image)
                image = image.to(device).unsqueeze(0)
                paras = model_hyper(image)
                # Building target network
                model_target = models.TargetNet(paras).to(device)
                for param in model_target.parameters():
                    param.requires_grad = False
                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
                score = float(pred.item())
                frame_scores.append(score)
        # split clip when 4 consecutive frames below threshold or frame with no detections
        clips = []
        curr_clip = []
        low_score_count = 0 # number of subpar frames
        for index, score in enumerate(frame_scores): # frame_scores[i] = score of track[i]
            track_items += 1
            if score < 0:
                low_score_count += 3
            elif score < frame_threshold: # count consecutive subpar frames
                low_score_count += 1
                track_buffer += score
            else:
                low_score_count = 0
                track_buffer = 0
            track_score += score
            if low_score_count > 4: # drop subpar frames -> split clip
                track_score -= track_buffer
                track_buffer = 0
                track_items -= 5
                if curr_clip:
                    clips.append(curr_clip)
                curr_clip = []
            else:
                curr_clip.append((score, (track[index])))
        if curr_clip:
            clips.append(curr_clip)
            
        # handle clips obtained from current track
        for clip in clips:
            scores = [score for score, img in clip]
            imgs = [img for score, img in clip]
            urls = [img.url for score, img in clip]
            clip_score = np.mean(scores)
            if clip_score >= clip_threshold:
                landmarks = get_all_landmarks(fa, imgs)
                m_clip = calculate_motion(landmarks)
                clip_score = alpha * clip_score + beta * m_clip
                final_clips.append((clip_score, urls))
    final_clips = sorted(final_clips, key=lambda x: x[0], reverse=True)
    final_urls = [url for score, url in final_clips]
    print(f"NUMBER OF FINAL TRACKS: {len(final_urls)}")
    save_tracks(final_urls, pickle_path)
    return final_urls