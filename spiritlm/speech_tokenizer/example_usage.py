#!/usr/bin/env python3
"""
Fixed example usage for the updated SpiritLM speech tokenizer with flexible audio handling.
"""

import IPython.display as ipd
import os
from spiritlm.config import set_global_checkpoint_dir
from spiritlm.speech_tokenizer import spiritlm_base

# Set up checkpoint directory (adjust path as needed for your environment)
# For Kaggle: set_global_checkpoint_dir("/kaggle/input/spiritlm-andinh")
# For local: set_global_checkpoint_dir("/path/to/your/spiritlm/checkpoints")
set_global_checkpoint_dir("/kaggle/input/spiritlm-andinh")  # Adjust this path

# Define your audio file path
audio = "/kaggle/input/sample-vimd/audio.flac"  # Your stereo 44.1kHz audio file

# Create tokenizer that preserves original sample rate and converts to mono
tokenizer = spiritlm_base(
    target_sample_rate=None,  # Keep original sample rate (44100 Hz in your case)
    auto_resample=True,
    preferred_channel='average'  # Convert stereo to mono by averaging
)

print(f'Processing audio: {audio}')
print('Original audio:')
ipd.display(ipd.Audio(audio))

## encode_units
print('\nSpiritLM-BASE: Encode audio into units (not deduplicated) \n', '-'*20)
units = tokenizer.encode_units(audio)
print(f"Units keys: {list(units.keys())}")
print(f"Hubert units (first 100 chars): {units['hubert'][:100]}...")
print(f"Total hubert units: {len(units['hubert'].split())} tokens")

## encode_string
print('\nSpiritLM-BASE: Encode audio into string (deduplicated and sorted units) \n', '-'*20)
string_tokens = tokenizer.encode_string(audio)
print(f"String tokens (first 200 chars): {string_tokens[:200]}...")
print(f"Total string length: {len(string_tokens)} characters")

## decode from units
print('\nSpiritLM-BASE: Decode back to audio from units (not deduplicated) \n', '-'*20)
resyn_wav = tokenizer.decode(units, speaker_id=1, dur_pred=False)
# Note: The output is always 16kHz regardless of input sample rate
ipd.display(ipd.Audio(resyn_wav, rate=16000))

## decode from string
print('\nSpiritLM-BASE: Decode back to audio from string (deduplicated and sorted units) \n', '-'*20)
resyn_dedup_wav = tokenizer.decode(string_tokens, speaker_id=1)
# Note: The output is always 16kHz regardless of input sample rate
ipd.display(ipd.Audio(resyn_dedup_wav, rate=16000))

# Additional: Show audio processing details
print('\n' + '='*50)
print('AUDIO PROCESSING DETAILS')
print('='*50)
print(f"Original audio file: {audio}")

# Load audio to show processing details
import torchaudio
wav, sr = torchaudio.load(audio)
print(f"Original shape: {wav.shape}")
print(f"Original sample rate: {sr} Hz")
print(f"Duration: {wav.shape[-1] / sr:.2f} seconds")
print(f"Channels: {wav.shape[0]}")

# Test with explicit sample rate method
print('\nTesting encode_units_with_sr method:')
units_explicit = tokenizer.encode_units_with_sr(wav, sr, channel_id=None)
print(f"Units from explicit method match: {units['hubert'] == units_explicit['hubert']}") 