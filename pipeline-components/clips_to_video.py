import os
import re
import moviepy.editor as mp


def get_keyframes(video_path):
    # Command to extract keyframes with timestamps
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time',
        '-read_intervals', f'%+{timestamp}%',
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
    return keyframes

def find_nearest_keyframe(keyframes, timestamp, forward=True):
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

def extract_frame_number(frame_url):
    print(frame_url)
    # Extract the frame number from the filename using regex
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        return int(match.group(1))
    return None

def read_status(json_file):
    if not os.path.exists(json_file):
        return {
            "total_videos_processed": 0,
            "video_duration_total": 0,
            "stage_2_total": 0,
            "stage_3_total": 0,
            "stage_4_total": 0,
            "final_clips_total": 0,
            "final_clips_duration_total": 0,
            "videos": {}
        }
    with open(json_file, 'r') as file:
        return json.load(file)

def write_status(json_file, status):
    with open(json_file, 'w') as file:
        json.dump(status, file, indent=4)
 
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

    # Update the specific video's parameter
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
    write_status(json_file, status)

def clips_to_videos(clips, extracted_fps, actual_fps, video_url, video_duration, output_base_path, video_name):
    clip_duration = 0
    # clips = [[clip1 jpgs], [clip2 jpgs], [clip3 jpgs], . . . ]
    video_name = os.path.basename(video_url).split('.')[0]
    video_output_folder = os.path.join(output_base_path, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    if os.listdir(video_output_folder):
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
        
        keyframes = get_keyframes(video_url)
        start_time = find_nearest_keyframe(keyframes, start_time, True) # forward = True
        end_time = find_nearest_keyframe(keyframes, end_time, False) # forward = False

        print(start_time)
        print(end_time)
        
        print(f"[START TIME, END TIME] = [{start_time}, {end_time}]")
        output_path = os.path.join(video_output_folder, f"clip{i}.mp4")

        command = (
            f'ffmpeg -y -ss {start_time} -i "{video_url}" -t {end_time - start_time} '
            f'-c copy "{output_path}"'
        )

        print(f"Running command: {command}")
        result = subprocess.run(command, capture_output=True, text=True, shell=True, encoding='utf-8')
        print(f"FFmpeg output: {result.stdout}")
        print(f"FFmpeg error: {result.stderr}")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg command failed with error: {result.stderr}")
        print(f"Saved clip to {output_path}")
    update_status('stats.json', video_name, 'final_clips_duration', clip_duration)

