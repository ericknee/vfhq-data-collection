# Stage 1 - Download YouTube videos
# Extract frames from videos
import cv2
import os
import shutil
import subprocess
import json

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
    
    timestamps = extract_frame_timestamps(video_path, desired_fps)
    frames = sorted(os.listdir(output_folder))
    frame_timestamps = {os.path.join(output_folder, frame): timestamps[i] for i, frame in enumerate(frames)}
    
    extracted_frame_count = len(os.listdir(output_folder))
    video_duration = get_video_duration(video_path)
    extracted_fps = extracted_frame_count / video_duration
    return video_folder, extracted_fps, original_fps, video_duration


video_folder = "videos"
for file_name in os.listdir(video_folder):
    if file_name.endswith(".mp4"):
        video_path = os.path.join(video_folder, file_name)
        try:
            print(f"Processing video: {video_path}")
            video_folder, extracted_fps, original_fps, video_duration, frame_timestamps = extract_frames(video_path)
            print(f"Finished processing video: {video_path}")
            print(f"Extracted FPS: {extracted_fps}")
            print(f"Original FPS: {original_fps}")
            print(f"Video Duration: {video_duration} seconds")
            for frame_path, timestamp in frame_timestamps.items():
                print(f"Frame: {frame_path}, Timestamp: {timestamp} seconds")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

