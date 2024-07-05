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

def get_landmarks(fa, image_path):
    image = cv2.imread(image_path)
    landmarks = fa.get_landmarks(image)
    if landmarks is not None:
        return landmarks[0]  # landmarks for first detected face
    return None

def get_all_landmarks(fa, frames):
    landmark_points = []
    for frame_path in frames:
        landmarks = get_landmarks(fa, frame_path)
        if landmarks is not None:
            landmark_points.append(landmarks)
            
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

def assess_clips(tracks, threshold=42, alpha=0.5, beta=0.2):
    iqa_scores = []
    final_clips = []
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cuda' if torch.cuda.is_available() else 'cpu', flip_input=False)
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_hyper = model_hyper.to(device)
    model_hyper.train(False)
    # load pre-trained model on koniq-10k dataset
    model_hyper.load_state_dict((torch.load('koniq_pretrained.pkl', map_location=device)))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])
    track_score = 0
    track_buffer = 0
    track_items = 0
    for track in tracks:
        frame_scores = [] # frames for each track
        for img in track:
            image = load_img(img.url)
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
        # split clip when 4 consecutive frames below threshold
        clips = []
        curr_clip = []
        low_score_count = 0 # number of subpar frames
        for index, score in enumerate(frame_scores): # frame_scores[i] = score of track[i]
            track_items += 1
            if score < threshold: # count consecutive subpar frames
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
                curr_clip.append((score, (track[index]).url))
        if curr_clip:
            clips.append(curr_clip)
            
        # handle clips obtained from current track
        for clip in clips:
            scores = [score for score, url in clip]
            urls = [url for score, url in clip]
            clip_score = np.mean(scores)
            if clip_score >= threshold:
                iqa_scores.append((clip_score, urls))
                landmarks = get_all_landmarks(fa, urls)
                m_clip = calculate_motion(landmarks)
                clip_score = alpha * clip_score + beta * m_clip
                final_clips.append((clip_score, urls))
    final_clips = sorted(final_clips, key=lambda x: x[0], reverse=True)
    final_urls = [url for score, url in final_clips[:3]]
    print(f"NUMBER OF FINAL TRACKS: {len(final_urls)}")
    return final_urls


directory_path = 'processed-videos/Robert Downey Jr Learns from Past Mistakes'
relative_file_paths = get_relative_file_paths(directory_path)
print("Relative file paths:\n", relative_file_paths)
start = time.time()
clips = assess_clips(relative_file_paths)
end = time.time()
print(clips)
print(f"Assess Clips Time - {end - start} seconds = {(end - start) / 60} minutes")