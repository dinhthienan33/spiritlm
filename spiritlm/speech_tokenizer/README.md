# Speech Tokenization for Spirit LM

This repo contains the speech encoder/decoder used for the Spirit LM.

Here is an example of how to use spiritlm_tokenizer

```python
import IPython.display as ipd
from spiritlm.speech_tokenizer import spiritlm_base, spiritlm_expressive

tokenizer = spiritlm_base() # base version, only has hubert units
# tokenizer = spiritlm_expressive() # expressive version, with pitch & style units

# Input audio
audio = "examples/audio/7143-88743-0029.flac"
print('Original audio:')
ipd.display(ipd.Audio(audio))

## encode_units
print('\nEncode audio into units (not deduplicated) \n', '-'*20)
units = tokenizer.encode_units(audio)
print(units)
# > {'audio': '.../audio/7143-88743-0029.flac', 'hubert': '99 49 38 149 149 71...'}

## encode_string
print('\nEncode audio into string (deduplicated and sorted units) \n', '-'*20)
string_tokens = tokenizer.encode_string(audio)
print(string_tokens)
# > '[Hu99][Hu49][Hu38][Hu149][Hu71]...'

## decode from units
print('\nDecode back to audio from units (not deduplicated) \n', '-'*20)
resyn_wav = tokenizer.decode(units, speaker_id=2, dur_pred=False)
ipd.display(ipd.Audio(resyn_wav, rate=16000))

## decode from string
print('\nDecode back to audio from string (deduplicated and sorted units) \n', '-'*20)
resyn_dedup_wav = tokenizer.decode(string_tokens, speaker_id=2)
ipd.display(ipd.Audio(resyn_dedup_wav, rate=16000))
```

## Flexible Audio Handling

The tokenizer now supports flexible audio handling for any sample rate and channel configuration:

```python
import torchaudio
from spiritlm.speech_tokenizer import spiritlm_base, spiritlm_expressive

# Example 1: Handle any sample rate automatically
tokenizer = spiritlm_base(
    target_sample_rate=None,  # Use original sample rate (no resampling)
    auto_resample=True,       # Automatically resample if target_sample_rate is set
    preferred_channel='average'  # Average all channels for multi-channel audio
)

# Example 2: Force 22kHz with specific channel selection
tokenizer_22k = spiritlm_expressive(
    target_sample_rate=22050,  # Resample everything to 22kHz
    auto_resample=True,
    preferred_channel=0,       # Use first channel for stereo/multi-channel audio
    max_wav_chunk_seconds=10,  # Process 10-second chunks
    min_wav_chunk_ms=100       # Minimum 100ms chunks
)

# Process audio with known sample rate
audio_array, sr = torchaudio.load("your_audio.wav")
units = tokenizer.encode_units_with_sr(audio_array, sr, channel_id=1)  # Use channel 1

# Process multi-channel audio
stereo_audio, sr = torchaudio.load("stereo.wav")  # Shape: (2, samples)
units_left = tokenizer.encode_units(stereo_audio, channel_id=0)   # Left channel
units_right = tokenizer.encode_units(stereo_audio, channel_id=1)  # Right channel
units_avg = tokenizer.encode_units(stereo_audio)  # Average both channels (default)

# Handle various sample rates
for audio_file in ["8khz.wav", "44khz.wav", "48khz.wav"]:
    units = tokenizer.encode_units(audio_file)  # Automatically handles different sample rates
```

### Parameters

- `target_sample_rate`: Target sample rate for processing (None = use original)
- `auto_resample`: Whether to automatically resample audio to target sample rate
- `preferred_channel`: Default channel handling for multi-channel audio (None, 0, 1, or 'average')
- `max_wav_chunk_seconds`: Maximum chunk duration in seconds for processing
- `min_wav_chunk_ms`: Minimum chunk duration in milliseconds

An example notebook can be found in [examples/speech_tokenizer/spiritlm_speech_tokenizer.ipynb](../../examples/speech_tokenizer/spiritlm_speech_tokenizer.ipynb).