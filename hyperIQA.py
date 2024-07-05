# Stage 4 - Selecting High Quality Clips
# hyperIQA to evaluate image quality
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import shutil
import models

def load_img(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def hyperIQA(track_path, output_path, threshold=42):
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_hyper = model_hyper.to(device)
    model_hyper.train(False)
    # load pre-trained model on koniq-10k dataset
    model_hyper.load_state_dict((torch.load('koniq_pretrained.pkl', map_location=device)))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

    for video_folder_entry in os.scandir(track_path):
        if video_folder_entry.is_dir():
            video_folder_path = os.path.join(track_path, video_folder_entry.name) # /content/drive/MyDrive/Creatify/tracks/vid_name
            for folder_entry in os.scandir(video_folder_path):
                if folder_entry.is_dir():
                    folder_path = os.path.join(video_folder_path, folder_entry.name)
                    track_score = 0
                    track_items = 0
                    if os.path.isdir(folder_path):
                        print(folder_path)
                        images = [os.path.join(folder_path, f.name) for f in os.scandir(folder_path) if f.is_file()]
                        if len(images) < 100:
                            continue
                        frame_scores = []
                        for img_path in images:
                            img = load_img(img_path)
                            img = transforms(img)
                            img = img.to(device).unsqueeze(0)
                            paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                            # Building target network
                            model_target = models.TargetNet(paras).to(device)
                            for param in model_target.parameters():
                                param.requires_grad = False

                            # Quality prediction
                            pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
                            score = float(pred.item())
                            frame_scores.append(score)

                        clips = []
                        curr_clip = []
                        low_score_count = 0
                        for score in frame_scores:
                            if score < threshold: # count consecutive subpar frames
                                low_score_count += 1
                            else:
                                low_score_count = 0

                            if low_score_count > 4: # drop subpar frames -> split clip
                                if curr_clip:
                                    clips.append(curr_clip)
                                curr_clip = []
                                low_score_count = 0
                            curr_clip.append(score)

                        if curr_clip:
                            clips.append(curr_clip)
                        start_index = 0
                        for idx, clip in enumerate(clips):
                            average_score = np.mean(clip)
                            if average_score >= threshold:
                                new_path = os.path.join(output_path, f"{folder_entry.name}_clip_{idx}")
                                os.makedirs(new_path, exist_ok=True)
                                clip_images = images[start_index:start_index + len(clip)]
                                for image in clip_images:
                                    shutil.copy(image, new_path)
                                print(f"COPIED - Clip {idx} of Track {folder_entry.name} score: {average_score}")
                            print(f"Clip {idx} of Track {folder_entry.name} score: {average_score}")
                            start_index += len(clip)

                        if (track_items != 0 and float(track_score) / float(track_items) >= threshold):
                            new_path = os.path.join(output_path, folder_entry.name)
                            os.makedirs(new_path, exist_ok=True)
                            for image in images:
                                shutil.copy(image, new_path)
                            print(f"COPIED - Track {folder_entry.name} score: {track_score / track_items}")
                        if (track_items != 0):
                          print(f"Track {folder_entry.name} - score: {track_score / track_items}")
                        else:
                          print(f"Track {folder_entry.name} - 0 track_items")

path = '/content/drive/MyDrive/Creatify/tracks'
output = '/content/drive/MyDrive/Creatify/hq-clips'
threshold = 42
hyperIQA(path, output, threshold)
