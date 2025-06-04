#!/usr/bin/env python3

import sys
sys.path.append('..')  # Adjust path as needed

import dask.dataframe as dd
import pandas as pd
import numpy as np
import logging
from vimd_processor import ViMDProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize_k_samples(k=5):
    """
    Tokenize only k samples from train dataset without computing entire dataframe
    
    Args:
        k (int): Number of samples to process
    """
    
    # Initialize processor (loads Hubert tokenizer)
    processor = ViMDProcessor()
    
    # Load train dataset path
    dataset_path = f"hf://datasets/nguyendv02/ViMD_Dataset/{processor.splits['train']}"
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Read parquet but don't compute yet
    df_dask = dd.read_parquet(dataset_path)
    
    # Get just the first k samples efficiently
    logger.info(f"Getting first {k} samples...")
    train_df_k = df_dask.head(k)  # This only computes k rows, not the entire dataset
    
    logger.info(f"Loaded {len(train_df_k)} samples")
    logger.info("Dataset columns:", train_df_k.columns.tolist())
    
    # Tokenize each sample
    results = []
    
    for i in range(len(train_df_k)):
        sample = train_df_k.iloc[i]
        audio_dict = sample['audio']
        
        print(f"\n--- Sample {i+1}/{k} ---")
        print(f"Filename: {sample['filename']}")
        print(f"Text: {sample['text']}")
        print(f"Speaker ID: {sample['speakerID']}")
        print(f"Audio bytes length: {len(audio_dict['bytes'])}")
        
        try:
            # Tokenize the sample
            unit_array, unit_string = processor.process_audio_to_tokens(audio_dict)
            
            print(f"✓ Tokenized successfully")
            print(f"  Number of tokens: {len(unit_array)}")
            print(f"  First 10 tokens: {unit_array[:10]}")
            print(f"  Unit string (first 50 chars): {unit_string[:50]}...")
            
            # Store results
            results.append({
                'filename': sample['filename'],
                'text': sample['text'],
                'speakerID': sample['speakerID'],
                'tokens_count': len(unit_array),
                'unit_array': unit_array,
                'unit_string': unit_string
            })
            
        except Exception as e:
            print(f"✗ Error tokenizing: {str(e)}")
            results.append({
                'filename': sample['filename'],
                'text': sample['text'],
                'speakerID': sample['speakerID'],
                'tokens_count': 0,
                'unit_array': np.array([]),
                'unit_string': ""
            })
    
    return results

def tokenize_k_samples_alternative(k=5):
    """
    Alternative method: Load dataset and take k samples using npartitions
    
    Args:
        k (int): Number of samples to process
    """
    
    # Initialize processor
    processor = ViMDProcessor()
    
    # Load train dataset 
    dataset_path = f"hf://datasets/nguyendv02/ViMD_Dataset/{processor.splits['train']}"
    df_dask = dd.read_parquet(dataset_path)
    
    # Get a small partition and take k samples
    logger.info(f"Getting {k} samples from first partition...")
    first_partition = df_dask.get_partition(0).compute()  # Compute only first partition
    
    # Take k samples from this partition
    k_samples = first_partition.head(k)
    
    logger.info(f"Loaded {len(k_samples)} samples from first partition")
    
    # Tokenize each sample
    for i in range(len(k_samples)):
        sample = k_samples.iloc[i]
        audio_dict = sample['audio']
        
        print(f"\n--- Sample {i+1}/{k} ---")
        print(f"Filename: {sample['filename']}")
        print(f"Text: {sample['text']}")
        
        try:
            unit_array, unit_string = processor.process_audio_to_tokens(audio_dict)
            print(f"✓ Tokenized - {len(unit_array)} tokens")
            print(f"  First 10 tokens: {unit_array[:10]}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    # Test with 3 samples
    k = 3
    print(f"=== Tokenizing {k} samples from train dataset ===")
    
    # Method 1: Using head() - most efficient
    results = tokenize_k_samples(k)
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed {len([r for r in results if r['tokens_count'] > 0])} out of {len(results)} samples")
    
    # Print token counts
    for i, result in enumerate(results):
        print(f"Sample {i+1}: {result['tokens_count']} tokens") 