import pickle
import re
import os

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


file_path = "processed-videos/Does Emily Blunt Know Her Lines From Her Most Famous Movies？/updated_tracks.pkl"
obj = load_tracks(file_path)
for track in obj:
    print(track)
print("hello")
        
