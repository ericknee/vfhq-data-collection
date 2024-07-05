#frames_to_video.py
import os
import cv2

def convert_frames_to_video(input_folder, output_video_base_path, fps=25):
    os.makedirs(output_video_base_path, exist_ok=True)
    for subfolder_entry in os.scandir(input_folder):
        if subfolder_entry.is_dir():
            subfolder_path = subfolder_entry.path
            frame_files = sorted([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith('.jpg')])

            if not frame_files:
                print(f"No jpg files found in {subfolder_path}. Skipping...")
                continue

            first_frame = cv2.imread(frame_files[0])  # get frame dimensions
            if first_frame is None:
                print(f"Could not read the first frame in {subfolder_path}. Skipping...")
                continue

            height, width, layers = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or other codec if needed
            output_video_path = os.path.join(output_video_base_path, f"{subfolder_entry.name}.mp4")
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    video_writer.write(frame)
                else:
                    print(f"Could not read frame {frame_file}. Skipping...")

            video_writer.release()
            print(f"Video saved to {output_video_path}")

# Example usage
input_folder = '/content/drive/MyDrive/Creatify/hq-clips'
output_video_base_path = '/content/drive/MyDrive/Creatify/final-clips'
convert_frames_to_video(input_folder, output_video_base_path)
