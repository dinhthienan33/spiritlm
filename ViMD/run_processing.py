#!/usr/bin/env python3
"""
Simple script to run ViMD dataset processing
"""

import argparse
import logging
from vimd_processor import ViMDProcessor, decode_from_csv

def main():
    parser = argparse.ArgumentParser(description="Process ViMD dataset with Hubert tokenizer")
    parser.add_argument("--action", choices=["process", "decode"], default="process",
                      help="Action to perform: process dataset or decode from CSV")
    parser.add_argument("--sample_fraction", type=float, default=0.01,
                      help="Fraction of data to sample (default: 0.01 for 1%)")
    parser.add_argument("--output_dir", default="processed_data",
                      help="Output directory for processed files")
    parser.add_argument("--csv_path", 
                      help="Path to CSV file for decoding (required for decode action)")
    parser.add_argument("--audio_output_dir", default="decoded_audio",
                      help="Output directory for decoded audio files")
    parser.add_argument("--splits", nargs="+", choices=["train", "test", "valid"], 
                      default=["train", "test", "valid"],
                      help="Dataset splits to process")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if args.action == "process":
        # Process dataset
        logger.info(f"Starting processing with {args.sample_fraction*100}% sampling")
        processor = ViMDProcessor(output_dir=args.output_dir)
        
        # Process specified splits
        for split_name in args.splits:
            try:
                logger.info(f"Processing {split_name} split...")
                processed_df = processor.process_split(split_name, args.sample_fraction)
                processor.save_to_csv(processed_df, split_name)
                logger.info(f"Successfully processed {split_name} split")
            except Exception as e:
                logger.error(f"Error processing {split_name}: {str(e)}")
        
        logger.info("Processing completed!")
        
    elif args.action == "decode":
        # Decode from CSV
        if not args.csv_path:
            logger.error("CSV path is required for decode action")
            return
        
        logger.info(f"Decoding audio from {args.csv_path}")
        success = decode_from_csv(args.csv_path, args.audio_output_dir)
        
        if success:
            logger.info("Decoding completed successfully!")
        else:
            logger.error("Decoding failed!")

if __name__ == "__main__":
    main() 