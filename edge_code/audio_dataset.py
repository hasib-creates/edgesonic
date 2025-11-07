import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class SimpleAudioDataset(Dataset):
    def __init__(
        self, 
        data_dir: Path,
        split: str = "train",
        audio_config: dict = None,
        transform=None,
        

    ):
        self.data_dir = Path(data_dir) / split
        self.audio_config = audio_config or {}
        self.transform = transform
        self.window_size = int(self.audio_config.get("target_length", 128)) 
        self.step_frames = int(self.audio_config.get("step_frames", self.window_size // 2)) 
        self.sliding = bool(self.audio_config.get("sliding_windows", True))

        self.files = []
        self.labels = []
        
        all_files = sorted([str(p) for p in Path(self.data_dir).rglob("*.wav")])
        
        for file_path in all_files:
            filename = Path(file_path).name.lower()
            
            if filename.startswith("normal_"):
                self.files.append(file_path)
                self.labels.append(0)
            elif filename.startswith("anomaly_"):
                self.files.append(file_path)
                self.labels.append(1)
            elif "normal" in filename:
                self.files.append(file_path)
                self.labels.append(0)
            elif "anomaly" in filename or "abnormal" in filename:
                self.files.append(file_path)
                self.labels.append(1)
            else:
                self.files.append(file_path)
                self.labels.append(0)
        
        normal_count = self.labels.count(0)
        anomaly_count = self.labels.count(1)
        self.has_anomalies = anomaly_count > 0
            
        print(f"Loaded {len(self.files)} files from {self.data_dir}")
        print(f"Normal: {normal_count}, Anomaly: {anomaly_count}")
        
        self._print_statistics()
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.audio_config.get("sample_rate", 16000),
            n_fft=self.audio_config.get("n_fft", 1024),
            hop_length=self.audio_config.get("hop_length", 512),
            n_mels=self.audio_config.get("num_mel_bins", 128),
        )
    
    def _print_statistics(self):
        source_normal = 0
        source_anomaly = 0
        target_normal = 0
        target_anomaly = 0
        
        for file_path, label in zip(self.files, self.labels):
            filename = Path(file_path).name.lower()
            is_source = 'source' in filename or 'target' not in filename
            is_normal = label == 0
            
            if is_source and is_normal:
                source_normal += 1
            elif is_source and not is_normal:
                source_anomaly += 1
            elif not is_source and is_normal:
                target_normal += 1
            else:
                target_anomaly += 1
        
        print(f"Dataset statistics for {self.data_dir.name}:")
        print(f"  Source Normal: {source_normal}")
        print(f"  Source Anomaly: {source_anomaly}")
        print(f"  Target Normal: {target_normal}")
        print(f"  Target Anomaly: {target_anomaly}")
        
        if hasattr(self, 'original_count'):
            print(f"  After oversampling: {len(self.files)} total samples")
        else:
            print(f"  Total samples: {len(self.files)}")
        
        print(f"  Has anomalies: {self.has_anomalies}")
        print(f"  Has multiple domains: {target_normal + target_anomaly > 0}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])

        target_sr = self.audio_config.get("sample_rate", 16000)
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if self.transform:
            waveform = self.transform(waveform)
            
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-6)

        mean = self.audio_config.get("norm_mean", -5.0)
        std = self.audio_config.get("norm_std", 4.5)
        mel_spec = (mel_spec - mean) / std

        target_length = self.audio_config.get("target_length", 128)
        if mel_spec.shape[-1] < target_length:
            mel_spec = torch.nn.functional.pad(
                mel_spec, (0, target_length - mel_spec.shape[-1])
            )
        else:
            mel_spec = mel_spec[..., :target_length]

        filename = Path(self.files[idx]).name.lower()
        domain = 1 if 'target' in filename else 0

        return mel_spec, domain, self.labels[idx]
    
    def get_normal_indices(self) -> List[int]:
        return [i for i, label in enumerate(self.labels) if label == 0]
    
    def get_anomaly_indices(self) -> List[int]:
        return [i for i, label in enumerate(self.labels) if label == 1]


def create_data_loaders(
    data_dir: Path,
    audio_config: dict,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = SimpleAudioDataset(
        data_dir=data_dir,
        split="train",
        audio_config=audio_config
    )
    
    test_dataset = SimpleAudioDataset(
        data_dir=data_dir,
        split="test",
        audio_config=audio_config
    )
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader