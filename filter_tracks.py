# Stage 2 - Face Tracking
# Filter tracks - keep tracks with frame lengths 200-2000

import os
import shutil

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


# Example usage
path = 'tracks'
filter_tracks(path)
