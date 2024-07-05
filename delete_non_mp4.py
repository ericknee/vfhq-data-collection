import os

def remove_non_mp4(folder_path):
    for file in os.scandir(folder_path):
        if file.is_file() and not file.name.endswith(".mp4"):
            os.remove(file.path)
            print(f"Removed: {file.path}")

from moviepy.editor import VideoFileClip

def delete_short_videos(folder_path, min_duration=300):
    for file in os.scandir(folder_path):
        if file.is_file() and file.name.endswith(".mp4"):
            try:
                video = VideoFileClip(file.path)
                duration = video.duration
                video.close()
                
                if duration < min_duration:
                    os.remove(file.path)
                    print(f"Removed {file.path}: Duration {duration} seconds")
            except Exception as e:
                print(f"Error processing {file.path}: {e}")

# Example usage
folder_path = "videos"
delete_short_videos(folder_path)

