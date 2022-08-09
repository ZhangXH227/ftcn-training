"""Classes for face forgery datasets (FaceForensics++, FaceShifter, DeeperForensics, Celeb-DF-v2, DFDC)"""

import bisect
from dataclasses import InitVar
import imp
import os
from tkinter.messagebox import NO

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import math

# dfdcroot='/public/zhaohanqing/workspace/DeepfakeDetectionBench/datasets/dfdc/'
ffpproot='/data/zhangxuehai/ffpp-faces/'
# deeperforensics_root='/public/zhaohanqing/workspace/DeepfakeDetectionBench/datasets/deeper/'
# fmfcc_root='/public/zhaohanqing/workspace/DeepfakeDetectionBench/datasets/fmfcc/'

class ForensicsClips_new32(Dataset):
    """Dataset class for FaceForensics++, FaceShifter, and DeeperForensics. Supports returning only a subset of forgery
    methods in dataset"""
    def __init__(
            self,
            real_videos,
            fake_videos,
            frames_per_clip,
            ds_types=['Origin', 'Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'],
            compression='c23',
            grayscale=False,
            transform=None,
            max_frames_per_video=270,
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        for ds_type in ds_types:

            # get list of video names
            video_paths = os.path.join('/data/zhangxuehai/ffpp-faces/', ds_type, compression, 'clips')
            if ds_type == 'Origin':
                videos = sorted(real_videos)
            elif ds_type == 'DeeperForensics':  # Extra processing for DeeperForensics videos due to naming differences
                videos = []
                for f in fake_videos:
                    for el in os.listdir(video_paths):
                        if el.startswith(f.split('_')[0]):
                            videos.append(el)
                videos = sorted(videos)
            else:
                videos = sorted(fake_videos)

            self.videos_per_type[ds_type] = len(videos)

            all_videos = os.listdir(video_paths)
            for video in videos:
                if not video in all_videos:
                    continue
                path = os.path.join(video_paths, video)
                num_clips = min(10, len(os.listdir(path)))
                self.clips_per_video.append(num_clips)
                self.paths.append([path, ds_type])

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
     
        item = self.paths[video_idx]
        path =  item[0]
        label =  0 if item[1] == 'Origin' else 1
        
        sample = []
        frames_path = os.path.join(path, str(clip_idx).zfill(4))
        frames = os.listdir(frames_path)
        n_holes = random.randint(1,3)
        lengths = []
        holes = []
        while 1:
            length0 = random.randint(1,math.floor(math.sqrt(0.8*224*224)))
            length1 = random.randint(1,math.floor(math.sqrt(0.8*224*224)))  if n_holes>1 else 0
            length2 = random.randint(1,math.floor(math.sqrt(0.8*224*224)))  if n_holes>2 else 0
            s_all = length0**2+length1**2+length2**2
            if s_all>0.2*224*224 and s_all<0.8*224*224:
                lengths.append(length0)
                lengths.append(length1)
                lengths.append(length2)
                break
        
        for n in range(n_holes):
            y = np.random.randint(224)
            x = np.random.randint(224)
            length = lengths[n]
            y1 = np.clip(y - length // 2, 0, 224)
            y2 = np.clip(y + length // 2, 0, 224)
            x1 = np.clip(x - length // 2, 0, 224)
            x2 = np.clip(x + length // 2, 0, 224)
            holes.append([y1,y2,x1,x2])
        for item in frames[:self.frames_per_clip]:
            with Image.open(os.path.join(frames_path, item)) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                if self.transform is not None:
                    img = self.transform(pil_img)
                    for n in range(n_holes):
                        mask = np.ones((224, 224), np.float32)
                        
                        y1,y2,x1,x2 = holes[n]
                        mask[y1: y2, x1: x2] = 0.
                        mask = torch.from_numpy(mask)
                        mask = mask.expand_as(img)
                        img = img * mask
                 
            sample.append(img)
        sample = torch.stack(sample,dim=1)
        
        return sample, frames_path.split('/')[-2], label

    def __getitem__(self, idx):
        try:
            sample, video_idx, label = self.get_clip(idx)
            label = torch.tensor(label, dtype=torch.long)

            return sample, label, video_idx
        except Exception as e:
            sample, video_idx, label = self.get_clip(idx+1)
            label = torch.tensor(label, dtype=torch.long)
            

            return sample, label, video_idx



import json
def load_json(name):
    with open(name) as f:
        a=json.load(f)
    return a


celebroot='/data/zhangxuehai/CelebDFv2_test_clips/'

class CelebDFClips(Dataset):
    """Dataset class for Celeb-DF-v2"""
    def __init__(
            self,
            frames_per_clip,
            grayscale=False,
            transform=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        Celeb_test = list(map(lambda x:[os.path.join(celebroot,x[0].split('/')[-1]),1-x[1]],load_json(celebroot+'celeb.json')))   # 518 videos
        
        ds_types = ['real', 'synthesis']
        self.videos_per_type[ds_types[0]], self.videos_per_type[ds_types[1]] = 0, 0
        all_videos = os.listdir(celebroot)
        for item in Celeb_test:
            if not item[0].split('/')[-1] in all_videos:
                    continue
            self.videos_per_type[ds_types[item[1]]] += 1
            path = item[0]
            label = item[1]
            num_clips = len(os.listdir(path))
            self.clips_per_video.append(num_clips)
            self.paths.append([path, label])

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return len(self.paths)

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        item = self.paths[video_idx]
        path = item[0]
        label = item[1]
        frames = sorted(os.listdir(path))

        sample = []
        frames_path = os.path.join(path, str(clip_idx).zfill(3))
        frames = os.listdir(frames_path)
        # if len(frames) != 32:
        #     print("clips error:", path)
        #     ss
        for item in frames[:self.frames_per_clip]:
            with Image.open(os.path.join(frames_path, item)) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                if self.transform is not None:
                    img = self.transform(pil_img)
                 
            sample.append(img)
        sample = torch.stack(sample,dim=1)

        return sample, video_idx, label

    def __getitem__(self, idx):
        sample, video_idx, label = self.get_clip(idx)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label, video_idx


class DFDCClips(Dataset):
    """Dataset class for DFDC"""
    def __init__(
            self,
            frames_per_clip,
            metadata,
            grayscale=False,
            transform=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.metadata = metadata
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        video_paths = os.path.join('./data', 'datasets', 'DFDC', 'cropped_mouths')
        videos = sorted(os.listdir(video_paths))
        for video in videos:
            path = os.path.join(video_paths, video)
            num_frames = len(os.listdir(path))
            num_clips = num_frames // frames_per_clip
            self.clips_per_video.append(num_clips)
            self.paths.append(path)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.paths[video_idx]
        video_name = path.split('/')[-1]
        frames = sorted(os.listdir(path))

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip

        sample = []
        for idx in range(start_idx, end_idx, 1):
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            sample.append(img)

        sample = np.stack(sample)

        return sample, video_idx, video_name

    def __getitem__(self, idx):
        sample, video_idx, video_name = self.get_clip(idx)

        label = self.metadata.loc[f'{video_name}.mp4']['is_fake']
        label = torch.tensor(label, dtype=torch.float32)

        sample = torch.from_numpy(sample).unsqueeze(-1)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, video_idx


if __name__ =="__main__":
    import torch
    from data.samplers import ConsecutiveClipSampler
    from torchvision.transforms import Compose, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
    from tqdm import tqdm
    import pandas as pd
    import utils

    train_split = pd.read_json('/data-x/g15/ffpp-faces/train.json', dtype=False)
    train_files_real, train_files_fake = utils.get_files_from_split(train_split)
    train_transform = Compose(
        [CenterCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    )

    real_data = ForensicsClips_new32(train_files_real,
        train_files_fake,
        32,
        grayscale=False,
        compression='c23',
        ds_types=["Origin"],
        transform=train_transform,
        max_frames_per_video=320,
    )
    fake_data = ForensicsClips_new32(train_files_real,
        train_files_fake,
        32,
        grayscale=False,
        compression='c23',
        ds_types=['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'],
        transform=train_transform,
        max_frames_per_video=320,
    )

    train_dataset = torch.utils.data.ConcatDataset([real_data, fake_data])
    validate_dataset = CelebDFClips(32, False, train_transform) 
    
    # train_sampler = ConsecutiveClipSampler(real_data.clips_per_video)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=train_sampler, num_workers=8, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(validate_dataset, batch_size=4, sampler=test_sampler, num_workers=8)
    print(f"=====>Totally {len(train_dataset)} train video clips...")
    print(f"=====>Totally {len(validate_dataset)} test videos...")

    for inputs, targets, index_vid in data_loader_train:
        print("Labels:", targets, "   ||    video index:", index_vid)

    for inputs, targets, index_vid in data_loader_val:
        print("Labels:", targets, "   ||    video index:", index_vid)
   
