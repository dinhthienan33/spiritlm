# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
from typing import Dict, List

import torchaudio

MOST_COMMON_STYLES = [71, 68, 98]


_logger = logging.getLogger(__name__)


def _toks_positions(toks: List[str], rate: float, dedup: bool):
    prev_tok = None
    res = []
    for i, tok in enumerate(toks):
        if (not dedup) or (prev_tok is None or tok != prev_tok):
            res += [(tok, i / rate)]
        prev_tok = tok
    return res


def units_to_string(
    units: Dict[str, str],
    has_pitch=False,
    has_style=False,
    hubert_rate=24.99,
    hubert_dedup=True,
    hubert_key="hubert",
    pitch_rate=12.5,
    pitch_dedup=True,
    pitch_key="pitch",
    style_rate=1,
    style_dedup=False,
    style_key="style",
) -> str:
    """
    Example:
     - input (units):
        {
            'hubert': '78 42 81 159 316 259',
            'pitch': '13 13 13 13 13 3',
            'style': '81 81 81 81 81 81',
        }
     - output:
        '[St81][Hu78][Pi13][Hu42][Hu81][Hu159][Hu316][Pi3][Hu259]'
    """

    combine_toks = []

    if has_style:
        combine_toks += _toks_positions(
            [f"[St{i}]" for i in units[style_key].split()], style_rate, style_dedup
        )
    if has_pitch:
        combine_toks += _toks_positions(
            [f"[Pi{i}]" for i in units[pitch_key].split()], pitch_rate, pitch_dedup
        )
    combine_toks += _toks_positions(
        [f"[Hu{i}]" for i in units[hubert_key].split()], hubert_rate, hubert_dedup
    )
    combine_toks = [tok_pos[0] for tok_pos in sorted(combine_toks, key=lambda x: x[1])]
    return "".join(combine_toks)


def get_random_most_common_style() -> int:
    return random.choice(MOST_COMMON_STYLES)


def string_to_units(
    gen,
    hubert_key="hubert",
    pitch_key="pitch",
    style_key="style",
    duplicate_hubert_for_multiple_pitch=False,
):
    """
    Convert from tokenized string to dictionary of units.
    The units are 'pre-duplicated' to match the number of hubert units.
    Examples
     - input:
        '[St81][Hu78][Pi13][Hu42][Hu81][Hu159][Hu316][Pi3][Hu259]'
     - output:
        {
            'hubert': '78 42 81 159 316 259',
            'pitch': '13 13 13 13 13 3',
            'style': '81 81 81 81 81 81',
        }
    """
    prev_hubert = None
    first_hubert = None
    prev_pitch = None
    first_pitch = None
    prev_style = None
    first_style = None
    prev_is_pitch = False  # If this is True, add prev_hubert to the codes
    hubert = []
    pitch = []
    style = []
    for item in gen.split("["):
        if item and len(item) > 2:
            if item.startswith("Hu") and item[2].isdigit():
                hubert += [item[2:-1]]
                pitch += [prev_pitch]
                style += [prev_style]
                prev_is_pitch = False
                prev_hubert = item[2:-1]
                if first_hubert is None:
                    first_hubert = item[2:-1]
            elif item.startswith("St") and item[2].isdigit():
                if prev_style is None:
                    first_style = item[2:-1]
                prev_style = item[2:-1]
            elif item.startswith("Pi") and item[2].isdigit():
                if duplicate_hubert_for_multiple_pitch and prev_is_pitch:
                    hubert += [prev_hubert]
                    pitch += [item[2:-1]]
                    style += [prev_style]
                if prev_pitch is None:
                    first_pitch = item[2:-1]
                prev_pitch = item[2:-1]
                prev_is_pitch = True
    if first_pitch is not None and first_style is None:
        # in rare case, style is not present, we select randomly a common style token to make decoding work
        first_style = str(get_random_most_common_style())
    for i in range(len(hubert)):
        if hubert[i] is None:
            hubert[i] = first_hubert
        if style[i] is None:
            style[i] = first_style
        if pitch[i] is None:
            pitch[i] = first_pitch
    units = {hubert_key: " ".join(hubert)}
    if first_pitch is not None:
        units[pitch_key] = " ".join(pitch)
    if first_style is not None:
        units[style_key] = " ".join(style)
    return units


class SpiritLMTokenizer:
    def __init__(
        self,
        hubert_model,
        pitch_model=None,
        style_model=None,
        hifigan_model=None,
        hubert_rate=24.99,
        hubert_dedup=True,
        hubert_key="hubert",
        pitch_rate=12.5,
        pitch_dedup=True,
        pitch_key="pitch",
        style_rate=1,
        style_dedup=False,
        style_key="style",
        target_sample_rate=None,  # Changed from expected_sample_rate, None means auto-detect
        max_wav_chunk_seconds=6.25,  # Changed from samples to seconds for flexibility
        min_wav_chunk_ms=80,  # Changed from samples to milliseconds for flexibility
        auto_resample=True,  # New parameter to control automatic resampling
        preferred_channel=None,  # New parameter for multi-channel handling
    ):
        super().__init__()

        self.hubert_model = hubert_model
        self.pitch_model = pitch_model
        self.style_model = style_model
        self.hifigan_model = hifigan_model

        self.hubert_rate = hubert_rate
        self.hubert_dedup = hubert_dedup
        self.hubert_key = hubert_key

        self.speech_token = "[Speech]"
        self.pitch_key = None
        self.style_key = None
        if pitch_model is not None:
            self.pitch_rate = pitch_rate
            self.pitch_dedup = pitch_dedup
            self.pitch_key = pitch_key
            if style_model is not None:
                self.style_rate = style_rate
                self.style_dedup = style_dedup
                self.style_key = style_key

        # Flexible audio handling parameters
        self.target_sample_rate = target_sample_rate  # None means use original sample rate
        self.max_wav_chunk_seconds = max_wav_chunk_seconds
        self.min_wav_chunk_ms = min_wav_chunk_ms
        self.auto_resample = auto_resample
        self.preferred_channel = preferred_channel  # None, 0, 1, or 'average'

    def _get_chunk_sizes(self, sample_rate):
        """Calculate chunk sizes based on sample rate and time-based parameters."""
        max_chunk = int(self.max_wav_chunk_seconds * sample_rate)
        min_chunk = int(self.min_wav_chunk_ms * sample_rate / 1000)
        return max_chunk, min_chunk

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        
        # Handle sample rate
        if self.target_sample_rate is not None and sr != self.target_sample_rate:
            if self.auto_resample:
                _logger.info(f"Resampling audio from {sr} Hz to {self.target_sample_rate} Hz")
                wav = torchaudio.functional.resample(
                    wav, orig_freq=sr, new_freq=self.target_sample_rate
                )
                sr = self.target_sample_rate
            else:
                _logger.warning(f"Audio sample rate {sr} Hz does not match target {self.target_sample_rate} Hz. Set auto_resample=True to automatically resample.")
        
        return wav, sr  # Return both wav and sample rate

    def encode_units(self, audio, channel_id=None):
        """
        Get the speech units in dictionary format, e.g.
        {
            'audio': 'path/to/audio.wav',
            'hubert': '1 1 2 2 3',
            'pitch': '15 15 20',
            'style': '7',
        }
        The audio can be the path to audio file or an array.
        For stereo audio file, channel_id can be set (0 or 1), or use preferred_channel from init.
        """
        units = {}
        sample_rate = None

        if isinstance(audio, str):
            units["audio"] = os.path.abspath(audio)
            audio, sample_rate = self.load_audio(audio)
        else:
            # If audio is already loaded, we need to determine the sample rate
            # Default to 16kHz if not specified and target_sample_rate is None
            if self.target_sample_rate is not None:
                sample_rate = self.target_sample_rate
            else:
                sample_rate = 16000  # Fallback default
                _logger.warning(f"Sample rate not provided for audio array. Using default {sample_rate} Hz. "
                              "Consider setting target_sample_rate or passing sample rate information.")
        
        audio = audio.squeeze()
        
        # Handle multi-channel audio with improved logic
        if len(audio.shape) == 2:
            num_channels = audio.shape[0]
            _logger.info(f"Found {num_channels}-channel audio with shape {audio.shape}")
            
            # Determine which channel to use
            channel_to_use = channel_id if channel_id is not None else self.preferred_channel
            
            if channel_to_use is None:
                # Default behavior: average all channels
                _logger.info("Averaging all channels. Set channel_id or preferred_channel to select specific channel.")
                audio = audio.mean(0)
            elif channel_to_use == 'average':
                _logger.info("Averaging all channels as requested.")
                audio = audio.mean(0)
            elif isinstance(channel_to_use, int):
                if 0 <= channel_to_use < num_channels:
                    _logger.info(f"Using channel {channel_to_use}")
                    audio = audio[channel_to_use]
                else:
                    _logger.warning(f"Channel {channel_to_use} not available. Audio has {num_channels} channels. Averaging instead.")
                    audio = audio.mean(0)
            else:
                _logger.warning(f"Invalid channel specification: {channel_to_use}. Averaging channels instead.")
                audio = audio.mean(0)
        
        assert len(audio.shape) == 1, f"Expected 1D audio after processing, got shape {audio.shape}"

        # Calculate chunk sizes based on actual sample rate
        max_wav_chunk, min_wav_chunk = self._get_chunk_sizes(sample_rate)
        
        hubert_units = []
        pitch_units = []
        style_units = []
        
        for start in range(0, len(audio), max_wav_chunk):
            audio_chunk = audio[start : start + max_wav_chunk]
            if len(audio_chunk) < min_wav_chunk:
                continue
            hubert_units.extend([str(i.item()) for i in self.hubert_model(audio_chunk)])
            if self.pitch_model is not None:
                pitch_units.extend(
                    [str(i.item()) for i in self.pitch_model(audio_chunk)]
                )
            if self.style_model is not None:
                style_units.extend(
                    [str(i.item()) for i in self.style_model(audio_chunk)]
                )

        units[self.hubert_key] = " ".join(hubert_units)
        if self.pitch_model is not None:
            units[self.pitch_key] = " ".join(pitch_units)
        if self.style_model is not None:
            units[self.style_key] = " ".join(style_units)
        return units

    def units2string(self, units):
        """
        Convert from dictionary of units to tokenized string.
        The units are (optionally deduped) sorted by time steps and interleaved
        """
        has_pitch = self.pitch_model is not None
        has_style = self.style_model is not None
        return units_to_string(
            units=units,
            has_pitch=has_pitch,
            has_style=has_style,
            hubert_rate=self.hubert_rate,
            hubert_dedup=self.hubert_dedup,
            hubert_key=self.hubert_key,
            pitch_rate=self.pitch_rate if has_pitch else None,
            pitch_dedup=self.pitch_dedup if has_pitch else None,
            pitch_key=self.pitch_key if has_pitch else None,
            style_rate=self.style_rate if has_style else None,
            style_dedup=self.style_dedup if has_style else None,
            style_key=self.style_key if has_style else None,
        )

    def encode_string(self, audio):
        """
        Tokenize the audio into string format, e.g.
        '[St7][Pi15][Hu1][Hu2][Pi20][Hu3]'
        """
        units = self.encode_units(audio)
        return self.units2string(units)

    def __call__(self, audio):
        """
        Default call method
        """
        return self.encode_string(audio)

    def string2units(self, gen, duplicate_hubert_for_multiple_pitch=False):
        """
        Convert from tokenized string to dictionary of units.
        The units are 'pre-duplicated' to match the number of hubert units.
        Examples
            - input:
                '[St81][Hu78][Pi13][Hu42][Hu81][Hu159][Hu316][Pi3][Hu259]'
            - output:
                {
                    'hubert': '78 42 81 159 316 259',
                    'pitch': '13 13 13 13 13 3',
                    'style': '81 81 81 81 81 81',
                }
        """
        return string_to_units(
            gen,
            hubert_key=self.hubert_key,
            pitch_key=self.pitch_key if self.pitch_key else "pitch",
            style_key=self.style_key if self.style_key else "style",
            duplicate_hubert_for_multiple_pitch=duplicate_hubert_for_multiple_pitch,
        )

    def decode(self, code, speaker_id=2, dur_pred=True):
        """
        code can be under text form ([Hu1][Hu2]) or units form ({'hubert': '1 2'})
        """

        assert self.hifigan_model is not None

        if isinstance(code, str):
            units = self.string2units(code)
        else:
            units = code

        # if units['hubert'] doesn't have the same number as units['f0']
        # then likely this is resynthesis task, and we'll set dur_pred=False
        if (
            self.pitch_key
            and self.pitch_key in units
            and len(units[self.pitch_key].split())
            != len(units[self.hubert_key].split())
        ):
            dur_pred = False

        wav = (
            self.hifigan_model(
                code=units[self.hubert_key],
                f0_code=(
                    units[self.pitch_key]
                    if self.pitch_key and self.pitch_key in units
                    else None
                ),
                style_code=(
                    units[self.style_key]
                    if self.style_key and self.style_key in units
                    else None
                ),
                dur_pred=dur_pred,
                speaker_id=speaker_id,
                not_dedup_code=True,
            )
            .detach()
            .cpu()
            .numpy()
        )

        return wav

    def encode_units_with_sr(self, audio, sample_rate, channel_id=None):
        """
        Get the speech units for audio array with explicit sample rate.
        This is useful when you already have loaded audio and know its sample rate.
        
        Args:
            audio: Audio tensor/array
            sample_rate: Sample rate of the audio
            channel_id: Channel to use for multi-channel audio (overrides preferred_channel)
        """
        # Store original target sample rate temporarily
        original_target_sr = self.target_sample_rate
        
        # Handle resampling if needed
        if self.target_sample_rate is not None and sample_rate != self.target_sample_rate:
            if self.auto_resample:
                _logger.info(f"Resampling audio from {sample_rate} Hz to {self.target_sample_rate} Hz")
                import torch
                if isinstance(audio, torch.Tensor):
                    audio = torchaudio.functional.resample(
                        audio, orig_freq=sample_rate, new_freq=self.target_sample_rate
                    )
                else:
                    # Convert to tensor, resample, then back to numpy if needed
                    import numpy as np
                    was_numpy = isinstance(audio, np.ndarray)
                    if was_numpy:
                        audio = torch.from_numpy(audio.copy())
                    audio = torchaudio.functional.resample(
                        audio, orig_freq=sample_rate, new_freq=self.target_sample_rate
                    )
                    if was_numpy:
                        audio = audio.numpy()
                sample_rate = self.target_sample_rate
            else:
                _logger.warning(f"Audio sample rate {sample_rate} Hz does not match target {self.target_sample_rate} Hz. Set auto_resample=True to automatically resample.")
        
        # Temporarily set target_sample_rate to current sample_rate for processing
        self.target_sample_rate = sample_rate
        
        try:
            # Process the audio
            return self.encode_units(audio, channel_id=channel_id)
        finally:
            # Restore original target sample rate
            self.target_sample_rate = original_target_sr

    def encode_string_with_sr(self, audio, sample_rate, channel_id=None):
        """
        Tokenize audio array with explicit sample rate into string format.
        
        Args:
            audio: Audio tensor/array
            sample_rate: Sample rate of the audio  
            channel_id: Channel to use for multi-channel audio
        """
        units = self.encode_units_with_sr(audio, sample_rate, channel_id)
        return self.units2string(units)
