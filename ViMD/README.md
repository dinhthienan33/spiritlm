# ViMD Dataset Processing with Hubert Tokenizer

This repository contains scripts to process the ViMD (Vietnamese Multi-Domain) dataset using the SpiritLM Hubert tokenizer. The scripts can:

1. Load data from Dask (with sampling)
2. Tokenize audio using Hubert tokenizer into units and strings
3. Save processed data as CSV files
4. Decode audio from CSV files

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have the SpiritLM checkpoints available. The scripts expect the checkpoints to be in the `../checkpoints/speech_tokenizer/` directory relative to the script location.

## Dataset Schema

### Input Dataset (ViMD)
- `region`: string[pyarrow] - Region information
- `province_code`: int64 - Province code
- `province_name`: string[pyarrow] - Province name
- `filename`: string[pyarrow] - Audio filename
- `text`: string[pyarrow] - Transcription text
- `speakerID`: string[pyarrow] - Speaker identifier
- `gender`: int64 - Speaker gender
- `audio`: object - Audio data in bytes format

### Output CSV Files
The processed CSV files contain:
- `region`: string[pyarrow] - Region information
- `province_code`: int64 - Province code
- `province_name`: string[pyarrow] - Province name
- `filename`: string[pyarrow] - Audio filename
- `text`: string[pyarrow] - Transcription text
- `speakerID`: string[pyarrow] - Speaker identifier
- `gender`: int64 - Speaker gender
- `base_unit`: string - Comma-separated Hubert unit tokens
- `base_string`: string - Space-separated Hubert unit string

## Usage

### 1. Process Dataset

Process all splits with 1% sampling (default):

```bash
python run_processing.py --action process
```

Process with custom sampling (e.g., 5%):

```bash
python run_processing.py --action process --sample_fraction 0.05
```

Process only specific splits:

```bash
python run_processing.py --action process --splits train valid
```

Custom output directory:

```bash
python run_processing.py --action process --output_dir my_processed_data
```

### 2. Decode Audio from CSV

Decode audio from processed CSV:

```bash
python run_processing.py --action decode --csv_path processed_data/train_processed.csv
```

With custom output directory:

```bash
python run_processing.py --action decode --csv_path processed_data/train_processed.csv --audio_output_dir decoded_train_audio
```

### 3. Using the Python API

```python
from vimd_processor import ViMDProcessor, decode_from_csv

# Create processor
processor = ViMDProcessor(output_dir="my_output")

# Process a single split
df = processor.process_split("train", sample_fraction=0.01)
processor.save_to_csv(df, "train")

# Decode from CSV
decode_from_csv("my_output/train_processed.csv", "decoded_audio")
```

## File Structure

```
ViMD/
├── vimd_processor.py      # Main processing class and functions
├── run_processing.py      # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── prompt.txt            # Original requirements
├── check_vimd.py         # Original exploration script
└── processed_data/       # Output directory (created automatically)
    ├── train_processed.csv
    ├── test_processed.csv
    └── valid_processed.csv
```

## Key Features

1. **Efficient Processing**: Uses Dask for large dataset handling
2. **Sampling Support**: Process only a fraction of data for testing
3. **Error Handling**: Robust error handling with detailed logging
4. **Flexible Output**: Save as CSV with proper handling of numpy arrays
5. **Audio Decoding**: Reconstruct audio from saved Hubert tokens
6. **Progress Tracking**: Shows progress bars during processing

## Notes

- Audio is automatically resampled to 16kHz mono for Hubert processing
- The Hubert tokenizer requires CUDA/GPU for optimal performance
- Large datasets may require significant memory and processing time
- CSV files store numpy arrays as comma-separated strings for compatibility

## Troubleshooting

1. **Memory Issues**: Reduce sample_fraction or process splits individually
2. **CUDA Errors**: Ensure proper PyTorch and CUDA installation
3. **Missing Checkpoints**: Verify SpiritLM checkpoint paths
4. **Network Issues**: Ensure stable internet for dataset downloading

## Example Output

After processing, you'll get CSV files with the following structure:

| region | province_code | province_name | filename | text | speakerID | gender | base_unit | base_string |
|--------|---------------|---------------|----------|------|-----------|--------|-----------|-------------|
| North | 1 | Ha Noi | audio1.wav | "Xin chào" | spk001 | 0 | "99,49,38,149" | "99 49 38 149" |

The `base_unit` column contains comma-separated Hubert tokens, and `base_string` contains space-separated tokens as used by the tokenizer. 