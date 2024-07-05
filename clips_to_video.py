import os
import re
import moviepy.editor as mp


def extract_frame_number(frame_url):
    print(frame_url)
    # Extract the frame number from the filename using regex
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        return int(match.group(1))
    return None

def clips_to_videos(clips, extracted_fps, actual_fps, video_url, output_base_path):
    print("CLIPS", clips)
    os.makedirs(output_base_path, exist_ok=True)

    for i, clip in enumerate(clips):
        frame_numbers = extract_frame_number(clip)
        if not frame_numbers:
            raise ValueError("No valid frame numbers extracted from the URLs.")

        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        
        start_time = start_frame / extracted_fps * actual_fps
        end_time = end_frame / extracted_fps * actual_fps
        
        video = mp.VideoFileClip(video_url)
        video_slice = video.subclip(start_time / actual_fps, end_time / actual_fps)
        
        output_path = os.path.join(output_base_path, f"clip{i}.mp4")
        video_slice.write_videofile(output_path, fps=actual_fps, codec="libx264")
