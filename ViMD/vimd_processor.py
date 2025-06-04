import dask.dataframe as dd
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import io
import os
from pathlib import Path
import logging
from tqdm import tqdm

# Import the spiritlm hubert tokenizer
import sys
sys.path.append('..')  # Adjust path as needed
from spiritlm.speech_tokenizer.hubert import spiritlm_hubert

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViMDProcessor:
    def __init__(self, output_dir="processed_data"):
        """
        Initialize the ViMD processor
        
        Args:
            output_dir (str): Directory to save processed CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize hubert tokenizer
        logger.info("Loading Hubert tokenizer...")
        self.hubert_tokenizer = spiritlm_hubert()
        logger.info("Hubert tokenizer loaded successfully")
        
        # Dataset splits configuration
        self.splits = {
            'train': 'data/train-*-of-*.parquet',
            'test': 'data/test-*-of-*.parquet',
            'valid': 'data/valid-*-of-*.parquet'
        }
    
    def load_dataset_split(self, split_name, sample_fraction=0.01):
        """
        Load a split of the ViMD dataset from dask
        
        Args:
            split_name (str): Name of the split ('train', 'test', 'valid')
            sample_fraction (float): Fraction of data to sample (default 1% = 0.01)
        
        Returns:
            dask.DataFrame: Loaded dataset split
        """
        logger.info(f"Loading {split_name} split...")
        dataset_path = f"hf://datasets/nguyendv02/ViMD_Dataset/{self.splits[split_name]}"
        df = dd.read_parquet(dataset_path)
        
        # Sample the specified fraction
        if sample_fraction < 1.0:
            logger.info(f"Sampling {sample_fraction*100}% of {split_name} split")
            df = df.sample(frac=sample_fraction)
        
        return df
    
    def process_audio_to_tokens(self, audio_dict):
        """
        Process audio bytes to hubert tokens
        
        Args:
            audio_dict (dict): Audio dictionary with 'bytes' key
        
        Returns:
            tuple: (base_unit_array, base_string)
        """
        try:
            # Extract audio from bytes
            audio_bytes = audio_dict['bytes']
            with io.BytesIO(audio_bytes) as f:
                # Load audio using librosa
                audio_array, sampling_rate = librosa.load(f, sr=16000, mono=True)  # Convert to 16kHz mono for hubert
            
            # Convert to torch tensor for hubert processing
            import torch
            audio_tensor = torch.from_numpy(audio_array).float()
            
            # Get hubert units using the tokenizer
            units = self.hubert_tokenizer.encode_units(audio_tensor)
            
            # Extract hubert units as array and string
            hubert_units_str = units['hubert']
            hubert_units_array = np.array([int(x) for x in hubert_units_str.split()])
            
            return hubert_units_array, hubert_units_str
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return np.array([]), ""
    
    def process_split(self, split_name, sample_fraction=0.01):
        """
        Process a complete dataset split
        
        Args:
            split_name (str): Name of the split to process
            sample_fraction (float): Fraction of data to sample
        
        Returns:
            pd.DataFrame: Processed dataframe
        """
        logger.info(f"Processing {split_name} split...")
        
        # Load the split
        df_dask = self.load_dataset_split(split_name, sample_fraction)
        
        # Convert to pandas for processing
        logger.info("Converting to pandas DataFrame...")
        df = df_dask.compute()
        
        logger.info(f"Processing {len(df)} samples...")
        
        # Initialize new columns
        base_units = []
        base_strings = []
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            try:
                unit_array, unit_string = self.process_audio_to_tokens(row['audio'])
                base_units.append(unit_array)
                base_strings.append(unit_string)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                base_units.append(np.array([]))
                base_strings.append("")
        
        # Create the final dataframe with required columns
        result_df = pd.DataFrame({
            'region': df['region'],
            'province_code': df['province_code'],
            'province_name': df['province_name'],
            'filename': df['filename'],
            'text': df['text'],
            'speakerID': df['speakerID'],
            'gender': df['gender'],
            'base_unit': base_units,
            'base_string': base_strings
        })
        
        return result_df
    
    def save_to_csv(self, df, split_name):
        """
        Save processed dataframe to CSV
        
        Args:
            df (pd.DataFrame): Processed dataframe
            split_name (str): Name of the split
        """
        output_path = self.output_dir / f"{split_name}_processed.csv"
        logger.info(f"Saving {split_name} split to {output_path}")
        
        # Note: We need to handle numpy arrays in CSV
        # Convert numpy arrays to string representation for CSV storage
        df_to_save = df.copy()
        df_to_save['base_unit'] = df_to_save['base_unit'].apply(lambda x: ','.join(map(str, x)) if len(x) > 0 else '')
        
        df_to_save.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} samples to {output_path}")
    
    def process_all_splits(self, sample_fraction=0.01):
        """
        Process all dataset splits
        
        Args:
            sample_fraction (float): Fraction of data to sample from each split
        """
        logger.info(f"Starting processing of all splits with {sample_fraction*100}% sampling...")
        
        for split_name in self.splits.keys():
            try:
                # Process the split
                processed_df = self.process_split(split_name, sample_fraction)
                
                # Save to CSV
                self.save_to_csv(processed_df, split_name)
                
                logger.info(f"Successfully processed {split_name} split")
                
            except Exception as e:
                logger.error(f"Error processing {split_name} split: {str(e)}")

    def load_k_samples(self, split_name, k=10):
        """
        Load only k samples from a dataset split efficiently
        
        Args:
            split_name (str): Name of the split ('train', 'test', 'valid')
            k (int): Number of samples to load
        
        Returns:
            pd.DataFrame: DataFrame with k samples
        """
        logger.info(f"Loading {k} samples from {split_name} split...")
        dataset_path = f"hf://datasets/nguyendv02/ViMD_Dataset/{self.splits[split_name]}"
        df_dask = dd.read_parquet(dataset_path)
        
        # Get just k samples efficiently using head()
        df_k = df_dask.head(k)
        
        logger.info(f"Successfully loaded {len(df_k)} samples from {split_name}")
        return df_k

def decode_from_csv(csv_path, output_audio_dir="decoded_audio"):
    """
    Function to decode audio from CSV file using the saved hubert units
    
    Args:
        csv_path (str): Path to the processed CSV file
        output_audio_dir (str): Directory to save decoded audio files
    
    Returns:
        bool: Success status
    """
    try:
        # Load the processed CSV
        df = pd.read_csv(csv_path)
        
        # Create output directory
        output_dir = Path(output_audio_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize spiritlm tokenizer with decoder
        from spiritlm.speech_tokenizer import spiritlm_base
        tokenizer = spiritlm_base()
        
        logger.info(f"Decoding {len(df)} samples from {csv_path}")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Decoding audio"):
            try:
                # Parse the base_unit string back to units
                if pd.notna(row['base_unit']) and row['base_unit'].strip():
                    unit_list = [int(x) for x in row['base_unit'].split(',') if x.strip()]
                    units_dict = {'hubert': ' '.join(map(str, unit_list))}
                    
                    # Decode to audio
                    decoded_wav = tokenizer.decode(units_dict)
                    
                    # Save the decoded audio
                    output_filename = f"{row['filename'].replace('.wav', '')}_decoded.wav"
                    output_path = output_dir / output_filename
                    
                    # Save as WAV file
                    sf.write(output_path, decoded_wav.squeeze(), 16000)
                    
                else:
                    logger.warning(f"Empty units for row {idx}, skipping...")
                    
            except Exception as e:
                logger.error(f"Error decoding row {idx}: {str(e)}")
        
        logger.info(f"Decoding completed. Audio files saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error in decode_from_csv: {str(e)}")
        return False

if __name__ == "__main__":
    # Create processor instance
    processor = ViMDProcessor(output_dir="processed_data")
    
    # Process all splits with 1% sampling
    processor.process_all_splits(sample_fraction=0.01)
    
    logger.info("Processing completed!")
    
    # Example of how to use the decode function
    # decode_from_csv("processed_data/train_processed.csv", "decoded_audio/train") 