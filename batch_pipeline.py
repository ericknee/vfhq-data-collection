import cv2
import os
from batch_face import RetinaFace
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
import gc
import math
from statistics import mode

def find_nearest_keyframe(video_path, timestamp, forward=True):
    if forward:
        interval = f"{timestamp}%+5"  # Search from the timestamp forward by 10 seconds
    else:
        interval = f"{timestamp-5}%{timestamp}"  # Search 10 seconds before the timestamp
    
    command = [ # extract keyframes w/ timestamp
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time',
        '-read_intervals', interval,
        '-print_format', 'json',
        '-skip_frame', 'nokey',
        video_path
    ]
    # Run the ffprobe command
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    
    # Parse the ffprobe output
    ffprobe_output = json.loads(result.stdout)
    
    if 'frames' not in ffprobe_output:
        raise ValueError("No keyframes found")

    keyframes = [float(frame['pkt_pts_time']) for frame in ffprobe_output['frames']]
    for keyframe in keyframes:
        print(keyframe)
    print(interval)
    # Find the nearest keyframe
    nearest_keyframe = None
    if forward:
        for keyframe in keyframes:
            if keyframe >= timestamp:
                nearest_keyframe = keyframe
                break
    else:
        for keyframe in reversed(keyframes):
            if keyframe <= timestamp:
                nearest_keyframe = keyframe
                break

    if nearest_keyframe is None:
        raise ValueError("No valid keyframe found in the specified direction")

    return nearest_keyframe

def read_status(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def write_status(json_file, status):
    with open(json_file, 'w') as file:
        json.dump(status, file, indent=4)

def update_traits(file, video_name, age, gender, race)
    status = read_status(json_file)
    videos = status["videos"]
    if video_name not in videos:
        status['total_videos'] += 1
    videos[video_name] = {
        'age': age,
        'gender': gender,
        'race': race
    }
    write_status(file, status)
    
def update_status(json_file, video_name, parameter_name, parameter_value):
    status = read_status(json_file)
    videos = status["videos"]

    if video_name not in videos:
        status["total_videos_processed"] += 1
        videos[video_name] = {
            "video_duration": 0,
            "stage_2": 0,
            "stage_3": 0,
            "stage_4": 0,
            "final_clips": 0,
            "final_clips_duration": 0
        }
    video_entry = videos[video_name]
    # Adjust totals before updating the video's parameter
    if parameter_name in video_entry:
        current_value = video_entry[parameter_name]
        status[f"{parameter_name}_total"] -= current_value
        video_entry[parameter_name] = parameter_value
        status[f"{parameter_name}_total"] += parameter_value
    else:
        video_entry[parameter_name] = parameter_value
        status[f"{parameter_name}_total"] += parameter_value

    # # If updating final_clips or video_duration, also adjust the totals
    # if parameter_name == "final_clips":
    #     status["final_clips_total"] += parameter_value - current_value
    # elif parameter_name == "video_duration":
    #     status["video_duration_total"] += parameter_value - current_value

    write_status(json_file, status)

def get_landmarks(fa, img, image):
    bbox = np.array([img.x1, img.y1, img.x2, img.y2])
    landmarks = fa.get_landmarks_from_image(image, detected_faces=[bbox]) # detected_faces = list of np.arrays
    if landmarks is not None:
        return landmarks[0]  # landmarks for first detected face
    return None

def get_all_landmarks(fa, images, batch_size=64):
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
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        frame_number = int(match.group(1))
        prefix = os.path.dirname(frame_url)
        return frame_number
    return None

def extract_prefix(frame_url):
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        frame_number = int(match.group(1))
        prefix = os.path.dirname(frame_url)
        return prefix
    return None

def construct_frame_url(prefix, frame_number):
    return os.path.join(prefix, f"frame_{frame_number:05d}.jpg")
        
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
        return video_folder, extracted_fps, original_fps, video_duration, video_name
    
    os.makedirs(output_folder, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={desired_fps}',
        os.path.join(output_folder, 'frame_%05d.jpg')
    ]

    subprocess.run(command, check=True)
    
    extracted_frame_count = len(os.listdir(output_folder))
    video_duration = get_video_duration(video_path)
    extracted_fps = extracted_frame_count / video_duration
    return video_folder, extracted_fps, original_fps, video_duration, video_name

def get_max_clips(video_duration, min_value=3, max_value=10):
    min_duration = 60         # 1 minute
    max_duration = 5400       # 90 minutes
    clamped_duration = max(min_duration, min(max_duration, video_duration))
    exp_duration = math.exp(clamped_duration / 5400)
    normalized_exp_duration = (exp_duration - math.exp(min_duration / 5400)) / (math.exp(max_duration / 5400) - math.exp(min_duration / 5400))
    result = min_value + normalized_exp_duration * (max_value - min_value)
    # ['0:3', '60:3', '300:3', '600:3', '1200:3', '1800:4', '2400:5', '3000:6', '3600:6', '4200:7', '4800:8', '5400:10']
    return int(result)

def sort_faces(video_folder, batch_size=128, scale_factor=0.15):  # video_folder = processed-videos/video
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

def filter_tracks(tracks, extracted_fps, min_duration=4, max_duration=300):
    new_tracks = defaultdict(list)
    additional_segments = []

    for track_num, track in tracks.items():
        frames = [extract_frame_number(img.url) for img in track if img is not None]
        min_frame = min(frames)
        max_frame = max(frames)
        track_length = max_frame - min_frame

        if track_length > max_duration * extracted_fps:
            num_segments = int(track_length // (max_duration * extracted_fps))
            remainder = track_length % (max_duration * extracted_fps)

            for i in range(num_segments):
                start_frame = min_frame + i * max_duration * extracted_fps
                end_frame = start_frame + max_duration * extracted_fps
                segment = [img for img in track if start_frame <= extract_frame_number(img.url) < end_frame]
                if i == 0:
                    new_tracks[track_num] = segment
                else:
                    additional_segments.append((track_num, segment))

            if remainder >= min_duration * extracted_fps:
                start_frame = min_frame + num_segments * max_duration * extracted_fps
                end_frame = max_frame
                segment = [img for img in track if start_frame <= extract_frame_number(img.url) < end_frame]
                additional_segments.append((track_num, segment))
        elif track_length >= min_duration * extracted_fps:
            new_tracks[track_num] = track

    # Add additional segments with new track numbers
    for i, (original_track_num, segment) in enumerate(additional_segments, start=1):
        new_track_num = f"{original_track_num}_{i}"
        new_tracks[new_track_num] = segment

    return new_tracks

def verify_faces(tracks, all_images, video_folder, threshold=1.24, min_frames=100, batch_size=64):
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
        # print("LENGTH", len(features), len(track))
        for i in range(1, len(features)):
            if features[i] is None or features[i - 1] is None:
                if len(current_id) >= min_frames: updated_tracks.append(current_id)
                current_id = []  # new identity track
                continue

            sim = l2_similarity(np.array(features[i - 1][0].get('embedding')), np.array(features[i][0].get('embedding')))
            print(sim, extract_frame_number(track[i - 1].url), extract_frame_number(track[i].url))
            if sim > threshold:
                # print(f"ARCFACE SPLIT: {track[i - 1].url} - {track[i].url}")
                if len(current_id) >= min_frames: updated_tracks.append(current_id)
                current_id = [track[i]]  # new identity track
            else:
                current_id.append(track[i])
        if len(current_id) > min_frames: updated_tracks.append(current_id)
    save_tracks(updated_tracks, pickle_path)
    return updated_tracks

def assess_clips(tracks, video_folder, frame_threshold=42, clip_threshold=45, alpha=0.5, beta=0.2, hyper_batch_size=64):
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
            elif score < frame_threshold or score < 0: # count consecutive subpar frames
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

def analyze_traits(clips, video_name):
    total_start = time.time()
    images = [cv2.imread(clip[0]) for clip in clips]
    age, gender, race = [], [], []
    for image in images:
        # print(clip)
        start = time.time()
        objs = DeepFace.analyze(
            img_path = image,
            actions = ['age', 'gender', 'race']
        )
        end = time.time()
        print("CLIP ANALYZE TIME:", end - start)
        age.append(objs[0]['age'])
        gender.append(objs[0]['dominant_gender'])
        race.append(objs[0]['dominant_race'])
        return objs
    update_status()
    print("Total analyzing time:", time.time() - total_start)

def clips_to_videos(clips, extracted_fps, actual_fps, video_url, video_duration, output_base_path, video_name):
    clip_duration = 0
    # clips = [[clip1 jpgs], [clip2 jpgs], [clip3 jpgs], . . . ]
    video_name = os.path.basename(video_url).split('.')[0]
    video_output_folder = os.path.join(output_base_path, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    if len(os.listdir(video_output_folder)) == len(clips):
        print("CLIPS ALREADY DOWNLOADED")
        return
    for i, clip in enumerate(clips):
        frame_numbers = [extract_frame_number(url) for url in clip]
        if not frame_numbers:
            raise ValueError("No valid frame numbers extracted from the URLs.")

        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        start_time = max(0, start_frame / extracted_fps)
        end_time = min(end_frame / extracted_fps, video_duration)
        clip_duration += end_time - start_time
        
        start_time = find_nearest_keyframe(video_url, start_time, True) # forward = True
        end_time = find_nearest_keyframe(video_url, end_time, False) # forward = False

        print(start_time)
        print(end_time)
        
        print(f"[START TIME, END TIME] = [{start_time}, {end_time}]")
        output_path = os.path.join(video_output_folder, f"{video_name}_clip{i}.mp4")

        command = (
            f'ffmpeg -y -ss {start_time} -i "{video_url}" -t {end_time} '
            f'-c copy -copyts "{output_path}"'
        )

        print(f"Running command: {command}")
        result = subprocess.run(command, capture_output=True, text=True, shell=True, encoding='utf-8')
        print(f"FFmpeg output: {result.stdout}")
        print(f"FFmpeg error: {result.stderr}")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg command failed with error: {result.stderr}")
        print(f"Saved clip to {output_path}")
    update_status('stats.json', video_name, 'final_clips_duration', clip_duration)

def main(video_path):
    # Stage 1 - Extract frames
    extract_start = time.time()
    proc_video_folder, extracted_fps, original_fps, video_duration, video_name = extract_frames(video_path)
    extract_end = time.time()
    print(f"Extract Frame Time: {extract_end - extract_start:2f}")
    update_status('stats.json', video_name, 'video_duration', video_duration)
    
    # Stage 2 - RetinaFace + SORT
    sort_start = time.time()
    tracks, all_images = sort_faces(proc_video_folder)
    sort_end = time.time()
    print(f"Bounding Boxes + Sort Time: {sort_end - sort_start:2f}")
    filtered_tracks = filter_tracks(tracks, extracted_fps)
    stage2_clips = len(filtered_tracks)
    update_status('stats.json', video_name, 'stage_2', stage2_clips)
    print(f"\n{stage2_clips} clips after stage 2\n")
    
    # Stage 3 - ArcFace
    verify_start = time.time()
    updated_tracks = verify_faces(filtered_tracks, all_images, proc_video_folder, min_frames=4 * extracted_fps)
    verify_end = time.time()
    print(f"ArcFace Verification Time: {verify_end - verify_start:2f}")
    stage3_clips = len(updated_tracks)
    update_status('stats.json', video_name, 'stage_3', stage3_clips)
    print(f"\n{stage3_clips} clips after stage 3\n")
    
    # Stage 4 - HyperIQA + Landmark Motion Tracking
    iqa_start = time.time()
    final_clips = assess_clips(updated_tracks, proc_video_folder)
    iqa_end = time.time()
    num_clips = get_max_clips(video_duration)
    top_clips = final_clips[:num_clips]
    print(f"Clip Quality Assessment Time: {iqa_end - iqa_start:2f}")
    stage4_clips = len(final_clips)
    update_status('stats.json', video_name, 'stage_4', stage4_clips)
    update_status('stats.json', video_name, 'final_clips', len(top_clips))
    print(f"\n{stage4_clips} clips after stage 4\n")
    
    del proc_video_folder, tracks, all_images, filtered_tracks, updated_tracks, final_clips
    gc.collect()
    
    analyze_traits(top_clips, video_name)
    clip_start = time.time()
    clips_to_videos(top_clips, extracted_fps, original_fps, video_path, video_duration, "final-clips", video_name)
    clip_end = time.time()
    print(f"Extract Frame Time: {extract_end - extract_start:2f}")
    print(f"Bounding Boxes + Sort Time: {sort_end - sort_start:2f}")
    print(f"ArcFace Verification Time: {verify_end - verify_start:2f}")
    print(f"Image Quality Assessment Time: {iqa_end - iqa_start:2f}")
    print(f"Frames To Video Time: {clip_end-clip_start:2f}")
    print(f"TOTAL RUNTIME: {clip_end - extract_start:2f}")

def process_directory(videos_dir='videos', processed_videos_dir='processed-videos'):
    for video_file in os.listdir(videos_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(videos_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            processed_video_folder = os.path.join(processed_videos_dir, video_name)
            frames_folder = os.path.join(processed_video_folder, 'frames')
            main(video_path)

# main("videos/Scarlett Johansson on Being a Movie Star vs. Being an Actor ｜ W Magazine.mp4")
# main("videos/DRAKE： Sundae Conversation with Caleb Pressley.mp4")
main("videos/Does Emily Blunt Know Her Lines From Her Most Famous Movies？.mp4")
# main("videos/THE GARFIELD MOVIE interviews with Chris Pratt, Samuel L Jackson, Jim Davis, director Mark Dindal 4K.mp4")
# main("videos/Open Thoughts with Kevin Hart.mp4")
# main('videos/Lilly Singh Fears for Her Life While Eating Spicy Wings ｜ Hot Ones.mp4')
# main('videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle.mp4')
# main('videos/A Smith Family Therapy Session ｜ Best Shape of My Life.mp4')
# main('videos/Why YouTube\'s Biggest Star Quit (Liza Koshy interview).mp4')
# main('videos/We Appraised JAY LENO\'s Car Collection (185 cars!).mp4')
# process_directory()