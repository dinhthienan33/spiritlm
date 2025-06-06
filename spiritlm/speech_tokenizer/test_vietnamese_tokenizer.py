#!/usr/bin/env python3
"""
Test script for trained Vietnamese Speech Tokenizer
Load and test the fine-tuned Vietnamese model

Usage:
    python test_vietnamese_tokenizer.py --model_dir ./vietnamese_tokenizer --audio_file test.wav
"""

import argparse
import json
import logging
from pathlib import Path
import torch
import torchaudio
import IPython.display as ipd

from spiritlm.config import set_global_checkpoint_dir
from spiritlm.speech_tokenizer import spiritlm_base
from spiritlm.speech_tokenizer.hubert.hubert_tokenizer import HubertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseTokenizer:
    """Vietnamese Speech Tokenizer with trained components"""
    
    def __init__(self, model_dir: str, pretrained_dir: str):
        self.model_dir = Path(model_dir)
        self.pretrained_dir = pretrained_dir
        
        # Set checkpoint directory
        set_global_checkpoint_dir(pretrained_dir)
        
        # Load configuration
        self.config = self._load_config()
        
        # Load base tokenizer
        self.tokenizer = spiritlm_base(
            target_sample_rate=self.config.get('sample_rate', 16000),
            auto_resample=True,
            preferred_channel='average'
        )
        
        # Load fine-tuned components
        self._load_vietnamese_components()
        
        logger.info("Vietnamese tokenizer loaded successfully")
        
    def _load_config(self):
        """Load training configuration"""
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No config file found, using defaults")
            return {'sample_rate': 16000}
            
    def _load_vietnamese_components(self):
        """Load fine-tuned Vietnamese components"""
        
        # Load fine-tuned quantizer if available
        quantizer_path = self.model_dir / "quantizer_vietnamese.pt"
        if quantizer_path.exists():
            logger.info("Loading Vietnamese quantizer...")
            state_dict = torch.load(quantizer_path, map_location='cpu')
            self.tokenizer.hubert_model.quantizer_model.load_state_dict(state_dict)
            logger.info("Vietnamese quantizer loaded")
        else:
            logger.warning("No Vietnamese quantizer found, using pretrained")
            
        # Load fine-tuned HiFiGAN if available
        hifigan_path = self.model_dir / "hifigan_vietnamese.pt"
        if hifigan_path.exists() and hasattr(self.tokenizer, 'hifigan_model'):
            logger.info("Loading Vietnamese HiFiGAN...")
            state_dict = torch.load(hifigan_path, map_location='cpu')
            self.tokenizer.hifigan_model.load_state_dict(state_dict)
            logger.info("Vietnamese HiFiGAN loaded")
        else:
            logger.info("No Vietnamese HiFiGAN found, using pretrained")
            
    def encode_audio(self, audio_path: str):
        """Encode Vietnamese audio to tokens"""
        logger.info(f"Encoding audio: {audio_path}")
        
        # Load and process audio
        wav, sr = torchaudio.load(audio_path)
        logger.info(f"Audio loaded: shape={wav.shape}, sr={sr}")
        
        # Encode to units
        units = self.tokenizer.encode_units_with_sr(wav, sr)
        
        # Encode to string
        string_tokens = self.tokenizer.units2string(units)
        
        return units, string_tokens
        
    def decode_audio(self, units_or_string, speaker_id: int = 0):
        """Decode tokens back to Vietnamese audio"""
        logger.info("Decoding tokens to audio...")
        
        # Decode using the (potentially Vietnamese-adapted) model
        reconstructed_wav = self.tokenizer.decode(
            units_or_string, 
            speaker_id=speaker_id, 
            dur_pred=True
        )
        
        return reconstructed_wav
        
    def test_audio_pipeline(self, audio_path: str, speaker_id: int = 0):
        """Test the complete encode-decode pipeline"""
        logger.info("Testing complete Vietnamese tokenizer pipeline...")
        
        # Original audio
        print("Original Vietnamese Audio:")
        ipd.display(ipd.Audio(audio_path))
        
        # Encode
        units, string_tokens = self.encode_audio(audio_path)
        
        print(f"\nTokenization Results:")
        print(f"- Hubert tokens: {len(units['hubert'].split())} tokens")
        print(f"- First 50 tokens: {units['hubert'][:100]}...")
        print(f"- String representation: {string_tokens[:200]}...")
        
        # Decode from units
        print("\nDecoded from units (non-deduplicated):")
        reconstructed_units = self.decode_audio(units, speaker_id)
        ipd.display(ipd.Audio(reconstructed_units, rate=16000))
        
        # Decode from string
        print("\nDecoded from string (deduplicated):")
        reconstructed_string = self.decode_audio(string_tokens, speaker_id)
        ipd.display(ipd.Audio(reconstructed_string, rate=16000))
        
        return units, string_tokens, reconstructed_units, reconstructed_string


def compare_models(audio_path: str, vietnamese_model_dir: str, pretrained_dir: str):
    """Compare original vs Vietnamese-adapted tokenizer"""
    
    print("="*60)
    print("COMPARING ORIGINAL VS VIETNAMESE-ADAPTED TOKENIZER")
    print("="*60)
    
    # Test original model
    print("\n1. ORIGINAL PRETRAINED MODEL:")
    print("-" * 40)
    set_global_checkpoint_dir(pretrained_dir)
    original_tokenizer = spiritlm_base(target_sample_rate=16000, preferred_channel='average')
    
    original_units = original_tokenizer.encode_units(audio_path)
    original_string = original_tokenizer.units2string(original_units)
    original_recon = original_tokenizer.decode(original_string, speaker_id=0)
    
    print(f"Original tokens: {len(original_units['hubert'].split())} tokens")
    print(f"Original string: {original_string[:100]}...")
    print("Original reconstruction:")
    ipd.display(ipd.Audio(original_recon, rate=16000))
    
    # Test Vietnamese model
    print("\n2. VIETNAMESE-ADAPTED MODEL:")
    print("-" * 40)
    vietnamese_tokenizer = VietnameseTokenizer(vietnamese_model_dir, pretrained_dir)
    
    vietnamese_units, vietnamese_string = vietnamese_tokenizer.encode_audio(audio_path)
    vietnamese_recon = vietnamese_tokenizer.decode_audio(vietnamese_string, speaker_id=0)
    
    print(f"Vietnamese tokens: {len(vietnamese_units['hubert'].split())} tokens")
    print(f"Vietnamese string: {vietnamese_string[:100]}...")
    print("Vietnamese reconstruction:")
    ipd.display(ipd.Audio(vietnamese_recon, rate=16000))
    
    # Compare tokenizations
    print("\n3. COMPARISON:")
    print("-" * 40)
    print(f"Token count difference: {len(vietnamese_units['hubert'].split()) - len(original_units['hubert'].split())}")
    
    # Simple similarity check
    original_tokens = set(original_units['hubert'].split())
    vietnamese_tokens = set(vietnamese_units['hubert'].split())
    common_tokens = original_tokens & vietnamese_tokens
    print(f"Common tokens: {len(common_tokens)}/{max(len(original_tokens), len(vietnamese_tokens))}")
    print(f"Token overlap: {len(common_tokens)/max(len(original_tokens), len(vietnamese_tokens))*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Test Vietnamese Speech Tokenizer')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to trained Vietnamese model')
    parser.add_argument('--pretrained_dir', type=str, default='/kaggle/input/spiritlm-andinh', help='Path to pretrained models')
    parser.add_argument('--audio_file', type=str, required=True, help='Vietnamese audio file to test')
    parser.add_argument('--speaker_id', type=int, default=0, help='Speaker ID for reconstruction')
    parser.add_argument('--compare', action='store_true', help='Compare with original model')
    
    args = parser.parse_args()
    
    # Test Vietnamese tokenizer
    vietnamese_tokenizer = VietnameseTokenizer(args.model_dir, args.pretrained_dir)
    vietnamese_tokenizer.test_audio_pipeline(args.audio_file, args.speaker_id)
    
    # Optional comparison
    if args.compare:
        compare_models(args.audio_file, args.model_dir, args.pretrained_dir)
    
    logger.info("Testing completed!")


if __name__ == '__main__':
    main() 