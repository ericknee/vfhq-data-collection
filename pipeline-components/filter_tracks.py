# Stage 2 - Face Tracking
# Filter tracks - keep tracks with frame lengths 200-2000

import os
import shutil

def filter_tracks(tracks):
    new_tracks = defaultdict(list)
    for track_num, track in tracks.items():
        frames = [extract_frame_number(img.url) for img in track if img is not None]
        min_frame = min(frames)
        max_frame = max(frames)
        if max_frame - min_frame > 2000:
            new_tracks[track_num] = track[:2000]
        elif max_frame - min_frame >= 100:
            # print(f"Keeping track {track_num} with length {max_frame - min_frame}")
            new_tracks[track_num] = track
    return new_tracks

path = 'tracks'
filter_tracks(path)
