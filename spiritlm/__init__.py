# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

# Import config to automatically override CHECKPOINT_DIR in other modules
from . import config

# Re-export commonly used items
from .config import BASE_CHECKPOINTS_DIR, SPIRITLM_MODEL_DIR, SPEECH_TOKENIZER_DIR, set_global_checkpoint_dir
