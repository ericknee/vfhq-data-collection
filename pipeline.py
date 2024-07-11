import cv2
import os
from retinaface import RetinaFace
import numpy as np
from sort import Sort
import time
from collections import defaultdict
from deepface import DeepFace
from PIL import Image
import tensorflow as tf
import torch
import torchvision
import models
import re
import moviepy.editor as mp
import face_alignment
import subprocess
import json
import pickle

def get_landmarks(fa, img):
    image = cv2.imread(img.url)
    bbox = np.array([img.x1, img.y1, img.x2, img.y2])
    landmarks = fa.get_landmarks_from_image(image, detected_faces=[bbox]) # detected_faces = list of np.arrays
    if landmarks is not None:
        return landmarks[0]  # landmarks for first detected face
    return None

def get_all_landmarks(fa, images):
    landmark_points = []
    for image in images:
        landmarks = get_landmarks(fa, image)
        if landmarks is not None:
            landmark_points.append(landmarks)
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

class Img:
    def __init__(self, url, x1, x2, y1, y2):
        self.url = url
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

def load_img(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def l2_similarity(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

def extract_frame_number(frame_url):
    # Extract the frame number from the filename using regex
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        return int(match.group(1))
    return None

def get_video_fps(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    
    ffprobe_output = json.loads(result.stdout)
    fps_str = ffprobe_output['streams'][0]['r_frame_rate']
    num, den = map(int, fps_str.split('/'))
    fps = num / den
    return fps

def save_tracks(tracks, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(tracks, f)
    print(f"Tracks saved to {file_path}")

def load_tracks(file_path):
    with open(file_path, 'rb') as f:
        tracks = pickle.load(f)
    print(f"Tracks loaded from {file_path}")
    return tracks

def get_video_duration(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    
    duration = float(result.stdout.strip())
    return duration

def extract_frame_timestamps(video_path, desired_fps):
    command = [
        'ffprobe',
        '-select_streams', 'v',
        '-show_frames',
        '-show_entries', 'frame=pts_time',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    frames = json.loads(result.stdout)['frames']
    timestamps = [float(frame['pts_time']) for frame in frames]
    return timestamps[::int(desired_fps)]

def extract_frames(video_path, output_base_folder="processed-videos", desired_fps=6):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_base_folder, video_name)
    print(f"Extracting frames from {video_name}")
    output_folder = os.path.join(video_folder, 'frames')
    original_fps = get_video_fps(video_path)
    if os.path.isdir(output_folder) and os.listdir(output_folder):
        extracted_frame_count = len(os.listdir(output_folder))
        video_duration = get_video_duration(video_path)
        extracted_fps = extracted_frame_count / video_duration
        print(f"FRAMES ALREADY EXTRACTED FOR {video_name}")
        return video_folder, extracted_fps, original_fps, video_duration
    
    os.makedirs(output_folder, exist_ok=True)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={desired_fps}',
        os.path.join(output_folder, 'frame_%05d.jpg')
    ]

    subprocess.run(command, check=True)
    
    # timestamps = extract_frame_timestamps(video_path, desired_fps)
    # frames = sorted(os.listdir(output_folder))
    # frame_timestamps = {os.path.join(output_folder, frame): timestamps[i] for i, frame in enumerate(frames)}
    
    extracted_frame_count = len(os.listdir(output_folder))
    video_duration = get_video_duration(video_path)
    extracted_fps = extracted_frame_count / video_duration
    return video_folder, extracted_fps, original_fps, video_duration

def sort_faces(video_folder): # video folder = processed-videos/video
    pickle_path = os.path.join(video_folder, 'tracks.pkl')
    input_folder = os.path.join(video_folder, 'frames')
    # if os.path.exists(pickle_path):
    #     return load_tracks(pickle_path)
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

            if detections is None or len(detections) > 1:
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
    # save_tracks(tracks, pickle_path)
    return tracks

def filter_tracks(tracks):
    for track_num in range(len(tracks) - 1, -1, -1):
        length = len(tracks[track_num])
        if length < 50:
            print(f"TRACK {track_num} DELETED\n LENGTH {length}")
            del tracks[track_num]
        elif length > 2000:
            tracks[track_num] = tracks[track_num][:2000]
            print(f"TRACK {track_num} TRUNCATED")
    return tracks

def verify_faces(tracks, video_folder, threshold=1.24, min_frames=50):
    pickle_path = os.path.join(video_folder, 'updated_tracks.pkl')
    if os.path.exists(pickle_path):
        return load_tracks(pickle_path)
    updated_tracks = []
    for track_num, track in tracks.items():
        if len(track) < min_frames:
            print(f"TRACK OF LENGTH {len(track)} SKIPPED")
            continue
        try:
            start_time = time.time()
            features = []
            for img in track:
                image = cv2.imread(img.url)
                if image is None:
                    print("NO IMAGE")
                    continue
                cropped_face = image
                face = Image.fromarray(cropped_face)
                embedding = DeepFace.represent(
                    img_path=np.array(face), 
                    model_name='ArcFace', 
                    enforce_detection=False,
                    detector_backend='skip'
                )
                features.append(embedding)
            # features = [DeepFace.represent(img_path=img.url, model_name='ArcFace', enforce_detection=False) for img in track]
            end_time = time.time()
            calc_time = end_time - start_time
            print(f"ArcFace Processing Time: {calc_time:2f} seconds")
        except Exception as e:
            print(f"Error processing images: {e}")

        current_id = [track[0]] # jpgs of same identity
        # print(len(features))
        for i in range(1, len(features)): # len(features) == len(track)
            if features[i] is None or features[i-1] is None:
                print("features[i] is None or features[i-1] is None")
                continue
            sim = l2_similarity(np.array(features[i-1][0].get('embedding')), np.array(features[i][0].get('embedding')))
            # print(i-1, i, "Similarity:", sim)
            if sim > threshold:
                print(f"ARCFACE SPLIT: {track[i-1].url} - {track[i].url}")
                if len(current_id) >= min_frames:
                    updated_tracks.append(current_id)
                current_id = [track[i]] # new identity track
            else:
                current_id.append(track[i])
        if len(current_id) > min_frames: updated_tracks.append(current_id)
    save_tracks(updated_tracks, pickle_path)
    return updated_tracks

def assess_clips(tracks, video_folder, threshold=42, alpha=0.5, beta=0.2):
    pickle_path = os.path.join(video_folder, 'urls.pkl')
    if os.path.exists(pickle_path):
        return load_tracks(pickle_path)
    iqa_scores = []
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
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])
    track_score = 0
    track_buffer = 0
    track_items = 0
    for track in tracks:
        frame_scores = [] # frames for each track
        for img in track: # img = Img object
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
                curr_clip.append((score, (track[index])))
        if curr_clip:
            clips.append(curr_clip)
            
        # handle clips obtained from current track
        for clip in clips:
            scores = [score for score, img in clip]
            imgs = [img for score, img in clip]
            print(f"\nLENGTH OF IMGS\n {len(imgs)}")
            urls = [img.url for score, img in clip]
            clip_score = np.mean(scores)
            if clip_score >= threshold:
                iqa_scores.append((clip_score, urls))
                landmarks = get_all_landmarks(fa, imgs)
                m_clip = calculate_motion(landmarks)
                clip_score = alpha * clip_score + beta * m_clip
                final_clips.append((clip_score, urls))
    final_clips = sorted(final_clips, key=lambda x: x[0], reverse=True)
    final_urls = [url for score, url in final_clips[:3]]
    print(f"NUMBER OF FINAL TRACKS: {len(final_urls)}")
    save_tracks(final_urls, pickle_path)
    return final_urls

def clips_to_videos(clips, extracted_fps, actual_fps, video_url, video_duration, output_base_path):
    # clips = [[clip1 jpgs], [clip2 jpgs], [clip3 jpgs], . . . ]
    video_name = os.path.basename(video_url).split('.')[0]
    video_output_folder = os.path.join(output_base_path, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    for i, clip in enumerate(clips):
        frame_numbers = [extract_frame_number(url) for url in clip]
        if not frame_numbers:
            raise ValueError("No valid frame numbers extracted from the URLs.")

        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        
        start_time = start_frame / extracted_fps
        start_time = max(0, start_time)
        end_time = end_frame / extracted_fps
        end_time = min(video_duration, end_time)
        print(f"[START TIME, END TIME] = [{start_time}, {end_time}]")
        output_path = os.path.join(video_output_folder, f"clip{i}.mp4")

        # command = f'ffmpeg -i "{video_url}" -ss {start_time} -to {end_time} -c copy "{output_path}"'
        command = (
            f'ffmpeg -ss {start_time} -to {end_time} -i "{video_url}" '
            # f'-vf "blackdetect=d=0.1:pic_th=0.98" '
            f'-c:v libx264 -y "{output_path}"'
        )
        print(f"Running command: {command}")
        result = subprocess.run(command, capture_output=True, text=True, shell=True, encoding='utf-8')
        print(f"FFmpeg output: {result.stdout}")
        print(f"FFmpeg error: {result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg command failed with error: {result.stderr}")

        print(f"Saved clip to {output_path}")

def main(video_path):
    # 11:14 minute video - 1099.239151 seconds = 18 minutes frame extraction time
    extract_start = time.time()
    proc_video_folder, extracted_fps, original_fps, video_duration = extract_frames(video_path)
    extract_end = time.time()
    print(f"Extract Frame Time: {extract_end - extract_start:2f}")
    
    sort_start = time.time()
    tracks = sort_faces(proc_video_folder)
    sort_end = time.time()
    print(f"Bounding Boxes + Sort Time: {sort_end - sort_start:2f}")
    # Bounding Boxes + Sort Time: 1565.495366
    # filter_start = time.time()
    # filtered_tracks = filter_tracks(tracks)
    # filter_end = time.time()
    # print(f"Filter Track Time: {filter_end - filter_start:2f}")
    # stage2_clips = len(filtered_tracks)
    # print(f"\n{stage2_clips} clips after stage 2\n")
    # for track in filtered_tracks:
    #     print(f"TRACK OF LENGTH {len(filtered_tracks[track])}")
        
    
    # verify_start = time.time()
    # updated_tracks = verify_faces(filtered_tracks, proc_video_folder)
    # verify_end = time.time()
    # print(f"ArcFace Verification Time: {verify_end - verify_start:2f}")
    # stage3_clips = len(updated_tracks)
    # print(f"\n{stage3_clips} clips after stage 3\n")
    
    # iqa_start = time.time()
    # final_clips = assess_clips(updated_tracks, proc_video_folder)
    # iqa_end = time.time()
    # print(f"Clip Quality Assessment Time: {iqa_end - iqa_start:2f}")
    # stage4_clips = len(final_clips)
    # print(f"\n{stage4_clips} clips after stage 4\n")
    
    # clip_start = time.time()
    # clips_to_videos(final_clips, extracted_fps, original_fps, video_path, video_duration, "final-clips")
    # clip_end = time.time()
    # print(f"Frames To Video Time: {clip_end-clip_start:2f}")
    
    # print(f"TOTAL RUNTIME: {clip_end - extract_start:2f}")
    # print(f"Extract Frame Time: {extract_end - extract_start:2f}")
    # print(f"Bounding Boxes + Sort Time: {sort_end - sort_start:2f}")
    # print(f"Filter Track Time: {filter_end - filter_start:2f}")
    # print(f"ArcFace Verification Time: {verify_end - verify_start:2f}")
    # print(f"HyperIQA Assessment Time: {iqa_end - iqa_start:2f}")
    # print(f"Frames To Video Time: {clip_end-clip_start:2f}")
    
# main("videos/Scarlett Johansson on Being a Movie Star vs. Being an Actor ｜ W Magazine.mp4")
main("videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle.mp4")