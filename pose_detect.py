import json
from posenet import SixDRepNet
from posenet import FaceDetector

import csv
import ffmpeg
import cv2
import inspect
import logging
import os
import tempfile
import argparse
from PIL import Image
from urllib.request import urlretrieve
import torch
import numpy as np
import subprocess
import glob
from timer import timer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import shlex

logger = logging.getLogger(__name__)
BUCKET_URL = "https://storage.googleapis.com/talking_face_data/crop_face"
root_dir = '/ytb-dl'
csv_file = os.path.join(root_dir, 'clips_v2', 'clips_info.csv')

import re
# import tempfile
# tempfile.tempdir = '/ytb-dl/tmp'

def remove_special_characters_and_replace_whitespace(filename):
    name, ext = os.path.splitext(filename)
    cleaned_name = ''.join(c for c in name if c.isalnum() or c.isspace() or c == '_')
    cleaned_name = re.sub(r'\s+', '_', cleaned_name)
    return f"{cleaned_name}{ext}"

class PoseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument(
            "--video_path",
            type=str,
            help="video path",
        )
        self.parser.add_argument(
            "--model_name", type=str, default="pose_model.pth", help=""
        )
        self.parser.add_argument(
            "--cache_dir",
            type=str,
            default="/tmp/.cache",
            help="cache dir to download model",
        )
        self.parser.add_argument("--cuda_id", type=int, default=0, help="0|1|2|...")
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="batch_size",
        )
        self.parser.add_argument(
            "--pose_thresh", type=float, default=20, help="pose degree to cut"
        )
        self.parser.add_argument(
            "--min_dur_length",
            type=float,
            default=30,
            help="minimum duration length is set to 30 seconds",
        )
        self.parser.add_argument(
            "--crop_model_name",
            type=str,
            default="sfd_face.pth",
            help="torch face crop",
        )

        return self.parser.parse_args()


def download_model(model_name, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    model_path = os.path.join(tgt_dir, model_name)
    if not os.path.exists(model_path):
        url = f"{BUCKET_URL}/{model_name}"
        # logging.info(f"download model from {url} to {model_path}")
        try:
            urlretrieve(url, model_path)
        except Exception as e:
            logging.error("Could not download file from {}".format(url))
    return model_path


class PoseDetect:
    def __init__(self, model_name, target_dir, cuda_id):
        self.model = SixDRepNet(
            gpu_id=cuda_id, dict_path=download_model(model_name, target_dir)
        )
        self.cuda_id = cuda_id

    def transform(self, imgs):
        imgs_ = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.model.transformations(img)
            imgs_.append(img)

        if len(imgs_) == 0:
            return None, None, None

        imgs_input = torch.stack(imgs_)
        imgs_input = imgs_input.cuda(self.cuda_id)
        return imgs_input

    def forward(self, frames):
        pitch, yaw, roll = self.model.batch_predict(frames)
        return {"p": pitch, "y": yaw, "r": roll}

    def forward_videos(self, video_path, batch_size=16):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        length = len(frames)
        step = length // batch_size + 1

        ps, ys, rs = [], [], []
        for i in range(step):
            cur_batch = frames[i * batch_size : (i + 1) * batch_size]
            p, y, r = self.model.batch_predict(cur_batch)

            ps.extend(p.tolist())
            ys.extend(y.tolist())
            rs.extend(r.tolist())

        meta_info = {"length": length, "ps": ps, "ys": ys, "rs": rs, "fps": fps}
        return meta_info

    def get_clips(self, meta_info, thresh):
        ps = meta_info["ps"]
        ys = meta_info["ys"]
        rs = meta_info["rs"]

        ps_ = [abs(a) for a in ps]
        ys_ = [abs(a) for a in ys]
        rs_ = [abs(a) for a in rs]

        idx_ps = np.array(ps_) < thresh
        idx_ys = np.array(ys_) < thresh
        idx_rs = np.array(rs_) < thresh

        idx = [x and y and z for x, y, z in zip(idx_ps, idx_ys, idx_rs)]

        b = np.r_[False, idx, False]
        s = np.flatnonzero(b[:-1] != b[1:])

        clips = []
        for start, end in zip(s[::2], s[1::2]):
            clips.append({"start_idx": start, "end_idx": end})
        
        return clips

    def get_optim_clips(self, meta_info, pose_thresh=20.0):
        clips = self.get_clips(meta_info, pose_thresh)

        if not clips:
            # logger.info(f"poses in all frames >= {pose_thresh}, omit.")
            return {"flag": False, "clips": []}

        ret_clips = []
        for clip in clips:
            start_idx = clip["start_idx"]
            end_idx = clip["end_idx"]
            ret_clips.append({
                "start_second_idx": start_idx,
                "end_second_idx": end_idx,
                # "start_time": start_idx / meta_info["fps"],
                # "end_time": end_idx / meta_info["fps"],
                "fps": meta_info["fps"],
            })

        return {"flag": True, "clips": ret_clips}


def extract_frames_from_video(video_path, save_dir):
    with timer("get video info"):
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        if int(fps) != 25:
            print(f"warning: the input video is {int(fps)}, != 25.")
        frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(save_dir, exist_ok=True)
    
    with timer("ffmpeg"):
        ffmpeg_command = [
            "ffmpeg",
            "-y",  # Add this flag to overwrite output files without asking
            "-i", str(video_path),  # Input file
            "-vf", "fps=1",  # Extract one frame per second
            os.path.join(save_dir, "%06d.png"),  # Output format for frames
        ]
        
        try:
            result = subprocess.run(
                ffmpeg_command,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                check=True,  # Raise an error if the command fails
            )
            print(result.stdout.decode())
            print(result.stderr.decode())
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg command failed with error: {e.stderr.decode()}")
            raise

    video_info = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps
    }
    return video_info

def load_image(path):
    return cv2.imread(path)[:, :, ::-1]

def save_clip_with_ffmpeg(video_path, start_time, end_time, clip_output_path):
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Add this flag to overwrite output files without asking
        "-i", video_path,  # Input file
        "-ss", str(start_time),  # Start time
        "-to", str(end_time),  # End time
        "-c", "copy",  # Copy codec
        clip_output_path  # Output file
    ]
    
    try:
        result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            check=True,  # Raise an error if the command fails
        )
        print(result.stdout.decode())
        print(result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed with error: {e.stderr.decode()}")
        raise

# def save_clip_with_ffmpeg(video_path, start_time, end_time, clip_output_path):
#     ffmpeg.input(video_path, ss=start_time, to=end_time).output(clip_output_path, codec='copy').run()


class Valid_Video_Detector:
    def __init__(self, opt):
        self.pd = PoseDetect(
            model_name=opt.model_name, target_dir=opt.cache_dir, cuda_id=opt.cuda_id
        )   

        fd_weight = download_model(opt.crop_model_name, opt.cache_dir)
        self.face_detector = FaceDetector(weight_file=fd_weight, device=opt.cuda_id)

        self.opt = opt

    def forward(self, video_path, min_dur_length=30, batch_size=4, pose_thresh=20.0):
        with tempfile.TemporaryDirectory() as tempdir:
            frame_dir = os.path.join(tempdir, "frames")
            os.makedirs(frame_dir, exist_ok=True)

            with timer("extract frames from video"):
                frame_meta = extract_frames_from_video(video_path, frame_dir)

            min_num_frames = min_dur_length
            # logger.info(f"min_num_frames = {min_num_frames}")

            video_frame_path_list = glob.glob(os.path.join(frame_dir, "*.png"))
            video_frame_path_list.sort()
            # logger.info("Length of video_frame_path_list:" + str(len(video_frame_path_list)))

            with timer("face detect"):
                num_frames = len(video_frame_path_list)
                step = num_frames // batch_size + 1

                crop_frames_clips, crop_frames = {}, []
                for i in range(step):
                    print(f"pose progress: {((i+1)/step*0.8):.2f}")
                    cur_batch = video_frame_path_list[
                        i * batch_size : (i + 1) * batch_size
                    ]
                    # logger.info(f"crop face {i}/{step}")
                    if len(cur_batch) == 0:
                        continue
                    with ThreadPoolExecutor(max_workers=len(cur_batch)) as executor:
                        frame_data_batch = np.array(
                            list(
                                executor.map(
                                    load_image,
                                    cur_batch,
                                )
                            )
                        )

                    crop_frame_data_batch = self.face_detector.batch_crop_img2(
                        frame_data_batch, width=224, height=224
                    )

                    # logger.info("Length of crop_frame_data_batch:" + str(len(crop_frame_data_batch)))

                    if len(crop_frame_data_batch) == 0:
                        if len(crop_frames) >= min_num_frames + 2:
                            # logger.info(
                            #     f"add clip crop face frames from idx: {i*batch_size-len(crop_frames)}, crop_frames.len = {len(crop_frames)}"
                            # )
                            crop_frames_clips[i * batch_size - len(crop_frames)] = (
                                crop_frames
                            )
                        crop_frames = []

                    crop_frames.extend(crop_frame_data_batch)

                    # logger.info("crop_frames:"+str(len(crop_frames)))

                if len(crop_frames) >= min_num_frames + 2:
                    # logger.info(
                    #     f"add clip crop face frames from idx: {i*batch_size-len(crop_frames)}, crop_frames.len = {len(crop_frames)}"
                    # )
                    crop_frames_clips[num_frames - len(crop_frames)] = crop_frames

            print("Length of crop_frames_clips:" + str(len(crop_frames_clips)))

            if len(crop_frames_clips) <= 0:
                ret_info = {"flag": False}
                return ret_info

            root_dir = '/ytb-dl'  # Update this path to your root directory
            output_dir = os.path.join(root_dir, 'clips_v2')
            print(f"output_dir:{output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            with timer("pose detect"):
                min_len = min_num_frames + 2
                frame_start, frame_end = 0, 0
                all_clips_info = []
                for offset, crop_frames in crop_frames_clips.items():
                    # logger.info(f"pose detect {offset}/{len(crop_frames)}/{num_frames}")
                    crop_length = len(crop_frames)
                    crop_step = crop_length // batch_size + 1

                    ps, ys, rs = [], [], []
                    for i in range(crop_step):
                        cur_batch = crop_frames[i * batch_size : (i + 1) * batch_size]
                        if len(cur_batch) == 0:
                            continue

                        p, y, r = self.pd.model.batch_predict(cur_batch)

                        ps.extend(p.tolist())
                        ys.extend(y.tolist())
                        rs.extend(r.tolist())

                    meta_info = {
                        "length": len(ps),
                        "ps": ps,
                        "ys": ys,
                        "rs": rs,
                        "fps": frame_meta["fps"],
                    }

                    ret_info = self.pd.get_optim_clips(meta_info, pose_thresh=pose_thresh)
                    # logger.info(f"ret_info:{ret_info}")
                    if ret_info["flag"]:
                        for clip in ret_info["clips"]:
                            if clip["end_second_idx"] - clip["start_second_idx"] >= min_len:
                                clip["start_second_idx"] += offset
                                clip["end_second_idx"] += offset
                                all_clips_info.append((clip, crop_frames[clip["start_second_idx"] - offset:clip["end_second_idx"] - offset]))
                                # logger.info(f"clip:{clip}")

                final_ret_info = {"flag": False, "clips": []}
                if all_clips_info:
                    print(f"Length of all_clips_info: {len(all_clips_info)}")
                    final_ret_info["flag"] = True
                    final_ret_info["clips"] = [clip_info[0] for clip_info in all_clips_info]

                    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                        futures = []
                        for idx, (clip, frames) in enumerate(all_clips_info):
                            video_file = os.path.basename(video_path)
                            filename = f'{os.path.splitext(video_file)[0]}_clip_{idx}.mp4'
                            cleaned_filename = remove_special_characters_and_replace_whitespace(filename)
                            print(f"filename:{cleaned_filename}")
                            clip_output_path = os.path.join(output_dir, cleaned_filename)

                            start_time = clip["start_second_idx"] + 1
                            end_time = clip["end_second_idx"] - 1

                            futures.append(executor.submit(save_clip_with_ffmpeg, video_path, start_time, end_time, clip_output_path))

                            clip["video_path"] = clip_output_path

                            with open(csv_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                file_name = filename
                                file_path = '/clips_v2/' + str(filename)
                                source = 'youtube'
                                is_training = 'true'
                                writer.writerow([file_name, file_path, source, is_training])
                        
                        for future in futures:
                            future.result()

            return final_ret_info

if __name__ == "__main__":
    opt = PoseOptions().parse_args()
    pipe = Valid_Video_Detector(opt)

    opt.pose_thresh = 20
    opt.min_dur_length = 4
    opt.batch_size = 1
    ret_info = pipe.forward(
        opt.video_path,
        min_dur_length=opt.min_dur_length,
        batch_size=opt.batch_size,
        pose_thresh=opt.pose_thresh,
    )
