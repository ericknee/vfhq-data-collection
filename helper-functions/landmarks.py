import argparse
import cv2
import torch
import torchlm
import os
import numpy as np
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tqdm import tqdm

def overlay_landmarks(frame, landmarks, color=(0, 0, 255), radius=2):
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame


def create_mask(
    valid_face,
    mask_file,
    frame_number,
    frame=None,
    landmarks=None,
    bboxes=None,
):
    if valid_face:
        x_from = int(landmarks[1][0])
        x_to = int(landmarks[15][0])
        y_from = int(landmarks[29][1])
        y_to = int(landmarks[8][1])
        # Write mask coordinates to file
        mask_file.write(f"{x_from}, {y_from}, {x_to}, {y_to}, {frame_number}\n")
        cv2.rectangle(
            frame, (x_from, y_from), (x_to, y_to), (0, 0, 0), thickness=cv2.FILLED
        )
        return frame
    else:
        mask_file.write(f"{-1}, {-1}, {-1}, {-1}, {frame_number}\n")


def save_frame(frame, frame_number, output_dir):
    frame_filename = os.path.join(output_dir, f"{frame_number:06d}.jpg")
    cv2.imwrite(frame_filename, frame)


def process_video(
    input_video_path,
    output_video_path,
    output_landmark_path,
    mask_coord_path,
    frame_output_dir,
    step,
    device,
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if step == 1:
        dinet_width, dinet_height = 80, 104
    elif step == 2:
        dinet_width, dinet_height = 160, 208
    elif step == 176:
        dinet_width, dinet_height = 220, 286
    elif step == 3:
        dinet_width, dinet_height = 320, 416
    else:
        dinet_width, dinet_height = 640, 832

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (dinet_width, dinet_height),
    )
    out_ldm = cv2.VideoWriter(
        output_landmark_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (dinet_width, dinet_height),
    )
    mask_file = open(mask_coord_path, "w")

    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18",
            pretrained=True,
            num_nb=10,
            num_lms=68,
            net_stride=32,
            input_size=256,
            meanface_type="300w",
            map_location=device,
        )
    )

    progress = tqdm(
        total=total_frames, desc=f"Processing {os.path.basename(input_video_path)}"
    )
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (dinet_width, dinet_height))
        # save_frame(frame, frame_number, frame_output_dir)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, bboxes = torchlm.runtime.forward(frame)

        if len(bboxes) > 0 and bboxes[0][4] > 0.97:
            landmarks_needed = landmarks[0][:68]
            # keep a buffer
            buffer = dinet_width // 16
            valid_landmarks = all(
                buffer < int(point[0]) < dinet_width - buffer
                and buffer < int(point[1]) < dinet_height - buffer
                for point in landmarks_needed
            )
            bboxes_needed = bboxes[0]
            valid_bbox = (
                buffer < int(bboxes_needed[0]) < dinet_width - buffer
                and buffer < int(bboxes_needed[2]) < dinet_width - buffer
                and buffer < int(bboxes_needed[1]) < dinet_height - buffer
                and buffer < int(bboxes_needed[3]) < dinet_height - buffer
            )
            valid_face = valid_landmarks and valid_bbox
            if valid_face:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                overlay_frame = overlay_landmarks(frame.copy(), landmarks_needed)
                masked_frame = create_mask(
                    valid_face,
                    mask_file,
                    frame_number,
                    frame,
                    landmarks_needed,
                    bboxes_needed,
                )
                out.write(masked_frame)
                out_ldm.write(overlay_frame)
            else:
                masked_frame = create_mask(valid_face, mask_file, frame_number)
        else:
            valid_face = False
            masked_frame = create_mask(valid_face, mask_file, frame_number)
            # out.write(frame)
        frame_number += 1
        progress.update(1)

    cap.release()
    out.release()
    out_ldm.release()
    mask_file.close()
    progress.close()


def process_directory(
    input_dir, output_dir, output_coords_dir, frame_output_dir, step, device
):
    for video_file in os.listdir(input_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(input_dir, video_file)
            output_video_path = os.path.join(output_dir, f"masked_{video_file}")
            output_landmark_path = os.path.join(output_dir, f"landmarks_{video_file}")
            mask_coord_path = os.path.join(
                output_coords_dir, f"{video_file}_mask_coords.txt"
            )
            process_video(
                video_path,
                output_video_path,
                output_landmark_path,
                mask_coord_path,
                frame_output_dir,
                step,
                device,
            )


if __name__ == "__main__":
    """
    python extract_video_landmarks.py \
    --input_dir '/app/data/dinet_creatify_all_new_crop/debug_oli' \
    --output_dir '/app/data/dinet_creatify_all_new_crop/landmarks_video_debug_oli12' \
    --output_coords_dir '/app/data/dinet_creatify_all_new_crop/landmarks_debug_oli12' \
    --step 3
    """
    parser = argparse.ArgumentParser(
        description="Process videos in a directory and overlay landmarks."
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing input video files"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save processed video files"
    )
    parser.add_argument(
        "--output_coords_dir",
        type=str,
        help="Directory to save processed landmarks coordination files",
    )
    parser.add_argument(
        "--frame_output_dir",
        type=str,
        default="/tmp",
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--step", type=int, default="4", help="Step to save processed video files"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for computation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.output_coords_dir):
        os.makedirs(args.output_coords_dir, exist_ok=True)

    if not os.path.exists(args.frame_output_dir):
        os.makedirs(args.frame_output_dir, exist_ok=True)

    process_directory(
        args.input_dir,
        args.output_dir,
        args.output_coords_dir,
        args.frame_output_dir,
        args.step,
        torch.device(args.device),
    )