# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

from .f0 import spiritlm_expressive_f0
from .hifigan import spiritlm_base_hifigan, spiritlm_expressive_hifigan_w2v2
from .hubert import spiritlm_hubert
from .spiritlm_tokenizer import SpiritLMTokenizer
from .style_encoder import spiritlm_expressive_style_encoder_w2v2

# Trick to avoid reloading the same model twice when calling multiple times
HUBERT = None
HIFIGAN_BASE = None
F0 = None
STYLE_W2V2 = None
HIFIGAN_EXPRESSIVE_W2V2 = None


def spiritlm_base(
    default_speaker=2,
    default_style=8,  # conv-default
    target_sample_rate=None,  # None means use original sample rate
    auto_resample=True,
    preferred_channel=None,  # None, 0, 1, or 'average'
    max_wav_chunk_seconds=6.25,
    min_wav_chunk_ms=80,
):
    # Hubert
    global HUBERT
    if HUBERT is None:
        HUBERT = spiritlm_hubert()

    # Hifigan
    global HIFIGAN_BASE
    if HIFIGAN_BASE is None:
        HIFIGAN_BASE = spiritlm_base_hifigan(
            default_speaker=default_speaker, default_style=default_style
        )

    return SpiritLMTokenizer(
        hubert_model=HUBERT,
        hifigan_model=HIFIGAN_BASE,
        target_sample_rate=target_sample_rate,
        auto_resample=auto_resample,
        preferred_channel=preferred_channel,
        max_wav_chunk_seconds=max_wav_chunk_seconds,
        min_wav_chunk_ms=min_wav_chunk_ms,
    )


def spiritlm_expressive(
    f0_backbone="fcpe", 
    default_speaker=2,
    target_sample_rate=None,  # None means use original sample rate
    auto_resample=True,
    preferred_channel=None,  # None, 0, 1, or 'average'
    max_wav_chunk_seconds=6.25,
    min_wav_chunk_ms=80,
):
    # Hubert
    global HUBERT
    if HUBERT is None:
        HUBERT = spiritlm_hubert()

    # F0
    global F0
    if F0 is None:
        F0 = spiritlm_expressive_f0(f0_backbone=f0_backbone)

    # Style
    global STYLE_W2V2
    if STYLE_W2V2 is None:
        STYLE_W2V2 = spiritlm_expressive_style_encoder_w2v2()

    # Hifigan
    global HIFIGAN_EXPRESSIVE_W2V2
    if HIFIGAN_EXPRESSIVE_W2V2 is None:
        HIFIGAN_EXPRESSIVE_W2V2 = spiritlm_expressive_hifigan_w2v2(
            default_speaker=default_speaker
        )

    return SpiritLMTokenizer(
        hubert_model=HUBERT,
        pitch_model=F0,
        style_model=STYLE_W2V2,
        hifigan_model=HIFIGAN_EXPRESSIVE_W2V2,
        hubert_key="hubert",
        pitch_key="pitch",
        style_key="style",
        target_sample_rate=target_sample_rate,
        auto_resample=auto_resample,
        preferred_channel=preferred_channel,
        max_wav_chunk_seconds=max_wav_chunk_seconds,
        min_wav_chunk_ms=min_wav_chunk_ms,
    )
