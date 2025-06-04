#!/usr/bin/env python3
"""
Test script to verify the setup and imports work correctly
"""

import sys
import logging

def test_imports():
    """Test if all required imports work"""
    try:
        import dask.dataframe as dd
        print("✓ Dask import successful")
    except ImportError as e:
        print(f"✗ Dask import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas import successful")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy import successful")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✓ SoundFile import successful")
    except ImportError as e:
        print(f"✗ SoundFile import failed: {e}")
        return False
    
    try:
        import librosa
        print("✓ Librosa import successful")
    except ImportError as e:
        print(f"✗ Librosa import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch import successful (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✓ CUDA available (device: {torch.cuda.get_device_name(0)})")
        else:
            print("! CUDA not available - will use CPU (slower)")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✓ TorchAudio import successful")
    except ImportError as e:
        print(f"✗ TorchAudio import failed: {e}")
        return False
    
    return True

def test_spiritlm_import():
    """Test SpiritLM imports"""
    try:
        # Add parent directory to path
        sys.path.append('..')
        
        from spiritlm.speech_tokenizer.hubert import spiritlm_hubert
        print("✓ SpiritLM Hubert import successful")
        
        from spiritlm.speech_tokenizer import spiritlm_base
        print("✓ SpiritLM base tokenizer import successful")
        
        return True
    except ImportError as e:
        print(f"✗ SpiritLM import failed: {e}")
        print("Make sure you're running this from the ViMD directory and SpiritLM is properly installed")
        return False
    except Exception as e:
        print(f"✗ SpiritLM setup failed: {e}")
        print("This might be due to missing checkpoints - check the checkpoint directory")
        return False

def test_dataset_access():
    """Test dataset access"""
    try:
        import dask.dataframe as dd
        test_path = "hf://datasets/nguyendv02/ViMD_Dataset/data/train-*-of-*.parquet"
        print("Testing dataset access...")
        
        # Just try to get the metadata without loading
        df = dd.read_parquet(test_path)
        print(f"✓ Dataset accessible (columns: {list(df.columns)})")
        return True
    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        print("This might be due to network issues or dataset access permissions")
        return False

def main():
    print("=== ViMD Processing Setup Test ===\n")
    
    print("1. Testing basic imports...")
    imports_ok = test_imports()
    print()
    
    print("2. Testing SpiritLM imports...")
    spiritlm_ok = test_spiritlm_import()
    print()
    
    print("3. Testing dataset access...")
    dataset_ok = test_dataset_access()
    print()
    
    print("=== Summary ===")
    if imports_ok and spiritlm_ok and dataset_ok:
        print("✓ All tests passed! You're ready to process the ViMD dataset.")
        print("\nTo start processing, run:")
        print("python run_processing.py --action process --sample_fraction 0.001  # Very small test")
    else:
        print("✗ Some tests failed. Please fix the issues above before proceeding.")
        if not imports_ok:
            print("  - Install missing dependencies with: pip install -r requirements.txt")
        if not spiritlm_ok:
            print("  - Make sure SpiritLM is properly installed and checkpoints are available")
        if not dataset_ok:
            print("  - Check your internet connection and dataset access permissions")

if __name__ == "__main__":
    main() 