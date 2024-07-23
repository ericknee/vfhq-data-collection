Usage

To process a single file: 
    python batch_pipeline.py --file path/to/video.mp4
    
To process all files in a directory:
    python batch_pipeline.py --directory path/to/videos

Output:
mp4 videos downloaded to processed_videos/final-clips

Stage 1 - Download 4K YouTube videos
Stage 2 - Detect bounding boxes with RetinaFace and create tracks with SORT algorithm
Stage 2.5 - Face ccclusion detection using YOLOv5 object detection
Stage 3 - Verify identities within tracks using ArcFace
Stage 3.5 - Cut detection with TransNetV2
Stage 4 - Assess clip quality with HyperIQA and landmark motion