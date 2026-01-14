import os
import torch
from torch.utils.data import Dataset
import torchvision
import pandas as pd
import numpy as np
from typing import Dict
import librosa


class SoundtrackDataset(Dataset):

    def __init__(
        self,
        path: str,
        train_or_valid
    ):
        # path: pasta onde estão os .csv e os .wav/.pt
        # ex: "./OpenScreenSoundLibrary-v1/"
        self.path = path
        self.video_fps = 25
        self.max_video_frames = 30 * self.video_fps

        # lê o CSV certo usando o path
        if train_or_valid == 'train':
            meta_path = os.path.join(self.path, "updated_meta.csv")
        elif train_or_valid == 'valid':
            meta_path = os.path.join(self.path, "updated_meta.csv")
        else:
            print("train or valid? - error")
            exit()

        self.data = pd.read_csv(meta_path)
        print(len(self.data))

    def __len__(self) -> int:
        return len(self.data)
    
    def normalize_audio(self, waveform, method='peak', eps=1e-10):
        # mantém compatível com o código antigo, mas
        # garante que o retorno seja tensor
        if isinstance(waveform, torch.Tensor):
            wf = waveform.detach().cpu().numpy()
        else:
            wf = np.array(waveform)
        
        if method == 'peak':
            peak = np.max(np.abs(wf))
            wf = wf / (peak + eps)
        
        elif method == 'rms':
            rms = np.sqrt(np.mean(wf**2))
            wf = wf / (rms + eps)
        
        elif method == 'minmax':
            max_val = np.max(wf)
            min_val = np.min(wf)
            wf = 2 * (wf - min_val) / (max_val - min_val + eps) - 1
        
        elif method == 'standard':
            mean = np.mean(wf)
            std = np.std(wf)
            wf = (wf - mean) / (std + eps)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # volta pra tensor com mesma dtype do original, se possível
        if isinstance(waveform, torch.Tensor):
            return torch.from_numpy(wf).type_as(waveform)
        return torch.from_numpy(wf).float()
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = librosa.load(audio_path, sr=32000)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # converte pra mono se tiver mais de um canal
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # corta ou preenche para 30s
        target_len = int(30 * sample_rate)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad_length = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return self.normalize_audio(waveform)
    
    def load_video(self, video_path: str) -> torch.Tensor:
        video, _, info = torchvision.io.read_video(
            video_path, 
            pts_unit='sec',
            output_format='TCHW'
        )
        
        if info['video_fps'] != self.video_fps:
            original_length = video.shape[0]
            target_length = int(original_length * self.video_fps / info['video_fps'])
            video = torch.nn.functional.interpolate(
                video.permute(1, 0, 2, 3),
                size=target_length,
                mode='linear'
            ).permute(1, 0, 2, 3)
        
        target_frames = 30 * self.video_fps
        if video.shape[0] > target_frames:
            video = video[:target_frames]
        else:
            pad_frames = target_frames - video.shape[0]
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_frames))
        
        return video

    def get_video_features(self, feat_path: str) -> torch.Tensor:
        return torch.load(feat_path)
    
    def generate_prompt(self, item):
        """
        Deixa 'mood' e 'caption' seguros:
        - se não existir, usa string vazia
        - se for NaN, vira string vazia
        - se for número (numpy.float64, int...), converte pra str
        """
        caption = item.get('caption', '')
        mood = item.get('mood', '')

        # trata NaN
        if pd.isna(caption):
            caption = ''
        if pd.isna(mood):
            mood = ''

        caption = str(caption)
        mood = str(mood)

        if mood:
            prompt = f"a film soundtrack for a {mood} scene. {caption}"
        else:
            prompt = f"a film soundtrack. {caption}"

        return prompt

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.data.iloc[idx]
        
        prompt = self.generate_prompt(item)
        file_name = f"{item['file_name']}"

        audio_path = os.path.join(self.path, file_name + ".wav")
        feat_path = os.path.join(self.path, file_name + ".pt")
    
        audio = self.load_audio(audio_path)
        video_features = self.get_video_features(feat_path)

        sample = {
            'prompt': prompt,
            'audio': audio,
            'video': video_features
        }
        
        return sample


class OESCom(Dataset):

    def __init__(self, csv_path, root):
        self.root = root
        self.data = pd.read_csv("./OpenScreenSoundLibrary-v1/updated_meta.csv")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def normalize_audio(self, waveform, method='peak', eps=1e-10):
        if isinstance(waveform, torch.Tensor):
            wf = waveform.detach().cpu().numpy()
        else:
            wf = np.array(waveform)
        
        if method == 'peak':
            peak = np.max(np.abs(wf))
            wf = wf / (peak + eps)
        
        elif method == 'rms':
            rms = np.sqrt(np.mean(wf**2))
            wf = wf / (rms + eps)
        
        elif method == 'minmax':
            max_val = np.max(wf)
            min_val = np.min(wf)
            wf = 2 * (wf - min_val) / (max_val - min_val + eps) - 1
        
        elif method == 'standard':
            mean = np.mean(wf)
            std = np.std(wf)
            wf = (wf - mean) / (std + eps)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if isinstance(waveform, torch.Tensor):
            return torch.from_numpy(wf).type_as(waveform)
        return torch.from_numpy(wf).float()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = librosa.load(audio_path, sr=32000)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return self.normalize_audio(waveform)

    
    def generate_prompt(self, item):
        caption = item.get('caption', '')
        mood = item.get('mood', '')

        if pd.isna(caption):
            caption = ''
        if pd.isna(mood):
            mood = ''

        caption = str(caption)
        mood = str(mood)

        if mood:
            prompt = f"a film soundtrack for a {mood} scene. {caption}"
        else:
            prompt = f"a film soundtrack. {caption}"

        return prompt

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.data.iloc[idx]

        film_id = item["film_id"]
        clip_id = item["clip_id"]

        base_name = f"{film_id}_{clip_id}"

        prompt = self.generate_prompt(item)
        
        audio_path = os.path.join(self.root, f"{base_name}.wav")
        video_path = os.path.join(self.root, f"{base_name}.pt")
        
        print("DEBUG audio_path:", audio_path)
        print("DEBUG exists:", os.path.exists(audio_path))
        print("DEBUG root exists:", os.path.exists(self.root))
        
        #audio_path = f"./OpenScreenSoundLibrary-v1/{idx}.wav"
        #video_path = f"./OpenScreenSoundLibrary-v1/{idx}.pt"

        audio = self.load_audio(audio_path)
        video_features = torch.load(video_path)

        sample = {
            'prompt': prompt,
            'audio': audio,
            'video': video_features
        }
        
        return sample
