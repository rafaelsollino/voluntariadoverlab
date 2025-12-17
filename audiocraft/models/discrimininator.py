import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
class ScaleDiscriminator(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,  # AudioCraft default
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(1, 1)),
            nn.Tanh()  # 또는 Sigmoid
        )
    def forward(self, audio):
        # Scale에 따른 다운샘플링
        if self.scale > 1:
            audio = F.avg_pool1d(audio.unsqueeze(1), 
                               kernel_size=self.scale, 
                               stride=self.scale).squeeze(1)
        
        # Mel spectrogram 변환
        mel = self.mel_transform(audio)
        mel = mel.unsqueeze(1)  # Add channel dimension
        
        # Log scale & Normalization
        mel = torch.log10(torch.clamp(mel, min=1e-5))
        mel = (mel + 5) / 5  # Normalize roughly to [-1, 1]
        
        # Discriminator
        feat = self.convs(mel)
        out = self.final(feat)
        
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(scale=s) for s in scales
        ])
    
    def forward(self, audio):
        return [d(audio) for d in self.discriminators]