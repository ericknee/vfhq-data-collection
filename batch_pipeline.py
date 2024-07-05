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
from batch_face import RetinaFace



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


def extract_frames(video_path, output_base_folder="processed-videos", desired_fps=6):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_base_folder, video_name)
    print(f"Extracting frames from {video_name}")
    os.makedirs(video_folder, exist_ok=True)
    output_folder = os.path.join(video_folder, 'frames')

    video_capture = cv2.VideoCapture(video_path)
    actual_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(actual_fps / desired_fps) if actual_fps > desired_fps else 1

    if os.path.isdir(output_folder) and os.listdir(output_folder):
        print(f"FRAMES ALREADY EXTRACTED FOR {video_name}")
        return output_folder, actual_fps
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    success, image = video_capture.read()
    while success:
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count // frame_interval:05d}.jpg")
            cv2.imwrite(frame_filename, image)
        success, image = video_capture.read()
        count += 1
    return output_folder, actual_fps

def sort_faces(input_folder, batch_size = 8): # video folder = processed-videos/video
    detector = RetinaFace(gpu_id=0)
    boxCount = 0
    tracks =  defaultdict(list)
    tracker = Sort()  # Initialize the SORT tracker
    files = sorted(os.scandir(input_folder), key=lambda e: e.name)
    for i in range(0, len(files), batch_size):
        file_names = [file.name for file in files[i:i+batch_size]]
        file_paths = [file.path for file in files[i:i+batch_size]]
        
        imgs = [cv2.imread(file_path) for file_path in file_paths] 
        start_time = time.time()
        faces = detector(imgs, cv=True)
        # print(faces)
        end_time = time.time()
        if len(faces) == 0: # no faces
            continue
        detection_time = end_time - start_time
        print(f"Detection time for batch {i / batch_size}: {detection_time:.2f} seconds")
        valid = False
        frame_detections = []  # Faces in the current frame
        # Process detections and prepare for tracking
        for face in faces:
            print(face)
            if len(face) == 0: continue
            box, landmarks, score = face
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width >= 500 and height >= 500:
                boxCount += 1
                print(boxCount)
                valid = True
                # SORT data format: [x1, y1, x2, y2, score]
                frame_detections.append([x1, y1, x2, y2, score])

            if valid:
                np_detections = np.array(frame_detections)
                tracked_objects = tracker.update(np_detections)

                for track in tracked_objects:
                    x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                    img = Img(file_paths[i], x1, x2, y1, y2)
                    tracks[track_id].append(img)
    return tracks

def filter_tracks(tracks):
    for track_num in range(len(tracks) - 1, -1, -1):
        length = len(tracks[track_num])
        if length < 10:
            print(f"TRACK {track_num} DELETED\n LENGTH {length}")
            del tracks[track_num]
        elif length > 2000:
            tracks[track_num] = tracks[track_num][:2000]
            print(f"TRACK {track_num} RUNCATED")
    return tracks

def verify_faces(tracks, threshold=1.5, min_frames=10):
    updated_tracks = []
    for track_num, track in tracks.items():
        if len(track) < min_frames:
            # print(f"TRACK OF LENGTH {len(track)} SKIPPED")
            continue
        track.sort(key=lambda img: img.url)
        try:
            start_time = time.time()
            features = []
            for img in track:
                image = cv2.imread(img.url)
                if image is None:
                    print("NO IMAGE")
                    continue
                cropped_face = image[img.y1:img.y2, img.x1:img.x2]
                face = Image.fromarray(cropped_face)
                embedding = DeepFace.represent(
                    img_path=np.array(face), 
                    model_name='ArcFace', 
                    enforce_detection=False, 
                    detector_backend='skip',
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
            print(i-1, i, "Similarity:", sim)
            if sim > threshold:
                print("sim > threshold")
                if len(current_id) >= min_frames:
                    updated_tracks.append(current_id)
                current_id = [track[i]] # new identity track
            else:
                current_id.append(track[i])
    return updated_tracks

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

def clips_to_videos(clips, extracted_fps, actual_fps, video_url, output_base_path):
    # clips = [[clip1 jpgs], [clip2 jpgs], [clip3 jpgs], . . . ]
    print("\nCLIPS\n", clips)
    os.makedirs(output_base_path, exist_ok=True)

    for i, clip in enumerate(clips):
        frame_numbers = [extract_frame_number(url) for url in clip]
        if not frame_numbers:
            raise ValueError("No valid frame numbers extracted from the URLs.")

        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        
        start_time = start_frame / extracted_fps * actual_fps
        end_time = end_frame / extracted_fps * actual_fps
        output_path = os.path.join(output_base_path, f"clip{i}.mp4")
        command = [
            "ffmpeg",
            "-i", video_url,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", "copy",
            # "-c:v", "libx264", 10 minutes frame to video
            # "-c:a", "aac",
            # "-strict", "experimental",
            output_path
        ]
        
        subprocess.run(command, check=True)
        print(f"Saved clip to {output_path}")

def main(video_path):
    # 11:14 minute video - 1099.239151 seconds = 18 minutes frame extraction time
    extract_start = time.time()
    frame_folder, actual_fps = extract_frames(video_path)
    extract_end = time.time()
    print(f"Extract Frame Time: {extract_end - extract_start:2f}")
    
    sort_start = time.time()
    tracks = sort_faces(frame_folder)
    sort_end = time.time()
    print(f"Bounding Boxes + Sort Time: {sort_end - sort_start:2f}")
    # Bounding Boxes + Sort Time: 1565.495366
    # filter_start = time.time()
    # filtered_tracks = filter_tracks(tracks)
    # filter_end = time.time()
    # print(f"Filter Track Time: {filter_end - filter_start:2f}")
    
    # verify_start = time.time()
    # updated_tracks = verify_faces(filtered_tracks)
    # verify_end = time.time()
    # print(f"ArcFace Verification Time: {verify_end - verify_start:2f}")
    
    # iqa_start = time.time()
    # final_clips = assess_clips(updated_tracks)
    # iqa_end = time.time()
    # print(f"HyperIQA Assessment Time: {iqa_end - iqa_start:2f}")
    
    # clip_start = time.time()
    # clips_to_videos(final_clips, 6, actual_fps, video_path, "final-clips")
    # clip_end = time.time()
    # print(f"Frames To Video Time: {clip_end-clip_start:2f}")
    
    # print(f"TOTAL RUNTIME: {clip_end - extract_start:2f}")
    # print(f"Extract Frame Time: {extract_end - extract_start:2f}")
    # print(f"Bounding Boxes + Sort Time: {sort_end - sort_start:2f}")
    # print(f"Filter Track Time: {filter_end - filter_start:2f}")
    # print(f"ArcFace Verification Time: {verify_end - verify_start:2f}")
    # print(f"HyperIQA Assessment Time: {iqa_end - iqa_start:2f}")
    # print(f"Frames To Video Time: {clip_end-clip_start:2f}")
    
# 2:54 Minutes = 174 seconds
# RetinaFace + SORT = 462.455001 seconds
main("videos/Robert Downey Jr Learns from Past Mistakes.mp4")

# test_url = "frame_00032.jpg"
# print(extract_frame_number(test_url))