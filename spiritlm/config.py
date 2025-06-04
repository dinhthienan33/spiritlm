# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

"""
Global configuration variables for SpiRit-LM.

This module contains global configuration variables that can be imported 
and used throughout the codebase.
"""

import os
import sys
from pathlib import Path

# Global base checkpoints directory - can be imported and used by any module
# Usage: from spiritlm.config import BASE_CHECKPOINTS_DIR
# Reads from environment variable SPIRITLM_CHECKPOINTS_DIR, defaults to /kaggle/input/speech-tokenizer
BASE_CHECKPOINTS_DIR = Path(os.environ.get('SPIRITLM_CHECKPOINTS_DIR', '/kaggle/input/speech-tokenizer'))

# Derived paths
SPIRITLM_MODEL_DIR = BASE_CHECKPOINTS_DIR / "spiritlm_model"
SPEECH_TOKENIZER_DIR = BASE_CHECKPOINTS_DIR / "speech_tokenizer"


def override_checkpoint_dirs():
    """
    Automatically override base_checkpoints_dir variables in ALL loaded modules.
    This function patches any module with base_checkpoints_dir without requiring code changes.
    """
    for module_name, module in sys.modules.items():
        if module:
            try:
                # Override base_checkpoints_dir if it exists in ANY module
                if hasattr(module, 'base_checkpoints_dir'):
                    setattr(module, 'base_checkpoints_dir', BASE_CHECKPOINTS_DIR)
                    print(f"✓ Overrode base_checkpoints_dir in {module_name}")
                    
            except Exception as e:
                print(f"⚠ Could not override base_checkpoints_dir in {module_name}: {e}")


# Automatically override checkpoint directories in other modules when this config is imported
override_checkpoint_dirs() 