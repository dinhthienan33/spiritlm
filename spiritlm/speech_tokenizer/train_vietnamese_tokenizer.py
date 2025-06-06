#!/usr/bin/env python3
"""
Training Script for Vietnamese Speech Tokenizer
Fine-tune pretrained SpiritLM tokenizer on Vietnamese speech dataset

Usage:
    python train_vietnamese_tokenizer.py --data_dir /path/to/vietnamese/speech --output_dir ./vietnamese_model
"""

import argparse
import logging
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from spiritlm.config import set_global_checkpoint_dir
from spiritlm.speech_tokenizer import spiritlm_base, spiritlm_expressive
from spiritlm.speech_tokenizer.hubert.hubert_tokenizer import HubertTokenizer
from spiritlm.speech_tokenizer.hifigan.hifigan_vocoder import HifiGANVocoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseSpeechDataset(Dataset):
    """Dataset for Vietnamese speech data"""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        max_duration: float = 10.0,
        min_duration: float = 1.0,
        target_sample_rate: int = 16000,
        audio_extensions: List[str] = [".wav", ".flac", ".mp3"]
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.target_sample_rate = target_sample_rate
        self.audio_extensions = audio_extensions
        
        # Load data
        self.audio_files = self._load_audio_files()
        logger.info(f"Loaded {len(self.audio_files)} audio files for {split}")
        
    def _load_audio_files(self) -> List[Path]:
        """Load all audio files from the data directory"""
        audio_files = []
        
        # Search for audio files
        for ext in self.audio_extensions:
            audio_files.extend(list(self.data_dir.rglob(f"*{ext}")))
            
        # Filter by duration if needed
        filtered_files = []
        for audio_file in audio_files:
            try:
                info = torchaudio.info(str(audio_file))
                duration = info.num_frames / info.sample_rate
                if self.min_duration <= duration <= self.max_duration:
                    filtered_files.append(audio_file)
            except Exception as e:
                logger.warning(f"Could not load {audio_file}: {e}")
                
        return filtered_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        audio_file = self.audio_files[idx]
        
        # Load audio
        try:
            wav, sr = torchaudio.load(str(audio_file))
            
            # Resample if needed
            if sr != self.target_sample_rate:
                wav = torchaudio.functional.resample(
                    wav, orig_freq=sr, new_freq=self.target_sample_rate
                )
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
                
            # Ensure minimum length
            min_samples = int(self.min_duration * self.target_sample_rate)
            if wav.shape[1] < min_samples:
                # Pad with zeros if too short
                padding = min_samples - wav.shape[1]
                wav = F.pad(wav, (0, padding))
            
            return {
                'audio': wav.squeeze(0),  # [T]
                'path': str(audio_file),
                'duration': wav.shape[1] / self.target_sample_rate
            }
            
        except Exception as e:
            logger.error(f"Error loading {audio_file}: {e}")
            # Return a dummy audio if loading fails
            dummy_audio = torch.zeros(int(2 * self.target_sample_rate))
            return {
                'audio': dummy_audio,
                'path': str(audio_file),
                'duration': 2.0
            }


class VietnameseTokenizerTrainer:
    """Main trainer class for Vietnamese speech tokenizer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set up checkpoints
        set_global_checkpoint_dir(config['pretrained_checkpoint_dir'])
        
        # Initialize models
        self._setup_models()
        
        # Setup data
        self._setup_data()
        
        # Setup training
        self._setup_training()
        
    def _setup_models(self):
        """Initialize pretrained models"""
        logger.info("Loading pretrained models...")
        
        # Load base tokenizer
        self.tokenizer = spiritlm_base(
            target_sample_rate=self.config['sample_rate'],
            auto_resample=True,
            preferred_channel='average'
        )
        
        # Extract individual models for fine-tuning
        self.hubert_model = self.tokenizer.hubert_model
        if hasattr(self.tokenizer, 'hifigan_model') and self.tokenizer.hifigan_model:
            self.hifigan_model = self.tokenizer.hifigan_model
        else:
            self.hifigan_model = None
            
        logger.info("Models loaded successfully")
        
    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        # Create datasets
        self.train_dataset = VietnameseSpeechDataset(
            data_dir=self.config['data_dir'],
            split='train',
            target_sample_rate=self.config['sample_rate'],
            max_duration=self.config['max_audio_duration'],
            min_duration=self.config['min_audio_duration']
        )
        
        # Create validation dataset if validation data exists
        val_data_dir = self.config.get('val_data_dir', None)
        if val_data_dir and Path(val_data_dir).exists():
            self.val_dataset = VietnameseSpeechDataset(
                data_dir=val_data_dir,
                split='val',
                target_sample_rate=self.config['sample_rate'],
                max_duration=self.config['max_audio_duration'],
                min_duration=self.config['min_audio_duration']
            )
        else:
            # Split train dataset
            train_size = int(0.9 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        
    def _setup_training(self):
        """Setup optimizers and schedulers"""
        
        # Determine what to train
        self.trainable_params = []
        
        if self.config['train_quantizer']:
            # Fine-tune the quantizer
            if hasattr(self.hubert_model, 'quantizer_model'):
                for param in self.hubert_model.quantizer_model.parameters():
                    param.requires_grad = True
                    self.trainable_params.append(param)
                logger.info("Quantizer will be fine-tuned")
                
        if self.config['train_hifigan'] and self.hifigan_model:
            # Fine-tune HiFiGAN
            for param in self.hifigan_model.parameters():
                param.requires_grad = True
                self.trainable_params.append(param)
            logger.info("HiFiGAN will be fine-tuned")
            
        if not self.trainable_params:
            logger.warning("No parameters to train! Enabling quantizer training by default.")
            for param in self.hubert_model.quantizer_model.parameters():
                param.requires_grad = True
                self.trainable_params.append(param)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        logger.info(f"Training {len(self.trainable_params)} parameters")
        
    def train_epoch(self) -> float:
        """Train one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        self.hubert_model.train()
        if self.hifigan_model and self.config['train_hifigan']:
            self.hifigan_model.train()
            
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move to device
            audio = batch['audio'].to(self.device)  # [B, T]
            
            loss = 0.0
            
            # Tokenize audio
            try:
                with torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad():
                    # Get tokens from audio
                    tokens = self.hubert_model(audio)  # [B, T_tokens]
                    
                    if self.config['train_quantizer']:
                        # Reconstruction loss for quantizer
                        # Get dense features
                        dense_features = self.hubert_model.get_dense_features(audio.unsqueeze(1))
                        
                        # Quantize and reconstruct
                        quantized_features = self.hubert_model.quantizer_model(dense_features)
                        
                        # Simple reconstruction loss
                        quantizer_loss = F.mse_loss(quantized_features, dense_features.detach())
                        loss += quantizer_loss
                        
                    if self.config['train_hifigan'] and self.hifigan_model:
                        # Reconstruction loss for HiFiGAN
                        try:
                            reconstructed_audio = self.hifigan_model(
                                code=' '.join([str(t.item()) for t in tokens[0]]),
                                speaker_id=0
                            )
                            
                            # Align lengths for loss computation
                            min_len = min(audio.shape[1], reconstructed_audio.shape[0])
                            audio_aligned = audio[0, :min_len]
                            recon_aligned = reconstructed_audio[:min_len]
                            
                            # Multi-scale spectral loss
                            recon_loss = self._compute_spectral_loss(audio_aligned, recon_aligned)
                            loss += recon_loss * self.config['reconstruction_weight']
                            
                        except Exception as e:
                            logger.debug(f"HiFiGAN reconstruction failed: {e}")
                            
                # Backward pass
                if loss.item() > 0:
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
            except Exception as e:
                logger.debug(f"Training step failed: {e}")
                continue
                
        return total_loss / max(num_batches, 1)
    
    def _compute_spectral_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale spectral loss"""
        loss = 0.0
        
        # Multiple STFT scales
        for hop_length in [256, 512, 1024]:
            n_fft = hop_length * 4
            
            # Compute spectrograms
            target_spec = torch.stft(
                target, n_fft=n_fft, hop_length=hop_length, 
                return_complex=True, normalized=True
            )
            pred_spec = torch.stft(
                prediction, n_fft=n_fft, hop_length=hop_length,
                return_complex=True, normalized=True
            )
            
            # Magnitude loss
            target_mag = torch.abs(target_spec)
            pred_mag = torch.abs(pred_spec)
            
            # Align shapes
            min_time = min(target_mag.shape[-1], pred_mag.shape[-1])
            target_mag = target_mag[..., :min_time]
            pred_mag = pred_mag[..., :min_time]
            
            loss += F.l1_loss(pred_mag, target_mag)
            
        return loss / 3.0  # Average over scales
        
    def validate(self) -> float:
        """Validate the model"""
        total_loss = 0.0
        num_batches = 0
        
        self.hubert_model.eval()
        if self.hifigan_model:
            self.hifigan_model.eval()
            
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                audio = batch['audio'].to(self.device)
                
                try:
                    # Simple validation: tokenize and measure consistency
                    tokens = self.hubert_model(audio)
                    
                    # Basic loss: ensure tokens are valid
                    valid_loss = 0.0
                    for token_seq in tokens:
                        # Check if tokens are in valid range
                        valid_tokens = (token_seq >= 0) & (token_seq < 500)  # Assume 500 vocab size
                        valid_loss += (1.0 - valid_tokens.float().mean())
                        
                    total_loss += valid_loss
                    num_batches += 1
                    
                except Exception as e:
                    logger.debug(f"Validation step failed: {e}")
                    continue
                    
        return total_loss / max(num_batches, 1)
    
    def save_model(self, epoch: int, output_dir: str):
        """Save the trained model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save quantizer if trained
        if self.config['train_quantizer']:
            quantizer_path = output_path / "quantizer_vietnamese.pt"
            torch.save(self.hubert_model.quantizer_model.state_dict(), quantizer_path)
            logger.info(f"Saved quantizer to {quantizer_path}")
            
        # Save HiFiGAN if trained
        if self.config['train_hifigan'] and self.hifigan_model:
            hifigan_path = output_path / "hifigan_vietnamese.pt"
            torch.save(self.hifigan_model.state_dict(), hifigan_path)
            logger.info(f"Saved HiFiGAN to {hifigan_path}")
            
        # Save config
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Model saved to {output_dir}")
        
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log progress
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, self.config['output_dir'])
                
            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = Path(self.config['output_dir']) / f"checkpoint_epoch_{epoch + 1}"
                self.save_model(epoch, str(checkpoint_dir))


def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            return json.load(f)


def create_default_config() -> Dict:
    """Create default training configuration"""
    return {
        # Data
        'data_dir': '/path/to/vietnamese/speech/data',
        'val_data_dir': None,  # If None, will split from train
        'sample_rate': 16000,
        'max_audio_duration': 10.0,
        'min_audio_duration': 1.0,
        
        # Model
        'pretrained_checkpoint_dir': '/kaggle/input/spiritlm-andinh',
        'train_quantizer': True,
        'train_hifigan': False,  # Start with quantizer only
        
        # Training
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'reconstruction_weight': 1.0,
        
        # System
        'num_workers': 4,
        'output_dir': './vietnamese_tokenizer',
        
        # Logging
        'log_interval': 100,
        'save_interval': 5
    }


def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese Speech Tokenizer')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Vietnamese speech data')
    parser.add_argument('--output_dir', type=str, default='./vietnamese_tokenizer', help='Output directory')
    parser.add_argument('--pretrained_dir', type=str, default='/kaggle/input/spiritlm-andinh', help='Pretrained model directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        
    # Override with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'pretrained_checkpoint_dir': args.pretrained_dir,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    })
    
    # Create trainer and start training
    trainer = VietnameseTokenizerTrainer(config)
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main() 