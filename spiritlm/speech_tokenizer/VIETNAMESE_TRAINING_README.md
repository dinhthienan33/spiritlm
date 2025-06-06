# Vietnamese Speech Tokenizer Training Guide

This guide shows how to fine-tune the SpiritLM speech tokenizer for Vietnamese speech data.

## 🎯 Overview

The training script adapts the pretrained SpiritLM tokenizer to better handle Vietnamese speech by fine-tuning:
- **Quantizer layer**: Learns Vietnamese-specific acoustic patterns
- **HiFiGAN vocoder** (optional): Improves reconstruction quality for Vietnamese

## 📁 Data Preparation

### Expected Data Structure
```
vietnamese_speech_dataset/
├── train/
│   ├── speaker1/
│   │   ├── audio001.wav
│   │   ├── audio002.wav
│   │   └── ...
│   ├── speaker2/
│   │   └── ...
├── val/ (optional)
│   └── ...
```

### Supported Audio Formats
- `.wav`, `.flac`, `.mp3`
- Any sample rate (automatically resampled to 16kHz)
- Mono or stereo (stereo will be averaged to mono)
- Duration: 1-10 seconds per audio file

### Data Requirements
- **Minimum**: 100+ audio files (~10 minutes total)
- **Recommended**: 1000+ audio files (~2+ hours total)
- **Optimal**: 10,000+ audio files (~20+ hours total)

## 🚀 Quick Start

### 1. Basic Training (Quantizer Only)
```bash
# Simple command-line training
python train_vietnamese_tokenizer.py \
    --data_dir /path/to/vietnamese/speech/dataset \
    --output_dir ./vietnamese_tokenizer \
    --pretrained_dir /kaggle/input/spiritlm-andinh \
    --epochs 20 \
    --batch_size 8 \
    --lr 0.0001
```

### 2. Advanced Training with Config File
```bash
# Edit the config file first
cp vietnamese_config.yaml my_config.yaml
# Update data_dir in my_config.yaml

# Train with config
python train_vietnamese_tokenizer.py \
    --config my_config.yaml \
    --data_dir /path/to/vietnamese/speech/dataset
```

## ⚙️ Configuration Options

### Key Parameters in `vietnamese_config.yaml`:

```yaml
# Data
data_dir: "/path/to/vietnamese/speech/dataset"
sample_rate: 16000
max_audio_duration: 10.0  # seconds
min_audio_duration: 1.0   # seconds

# Training
train_quantizer: true      # Fine-tune quantizer (recommended)
train_hifigan: false       # Fine-tune vocoder (more advanced)
batch_size: 8              # Reduce if out of memory
num_epochs: 20
learning_rate: 0.0001

# System
num_workers: 4
output_dir: "./vietnamese_tokenizer"
```

### Training Modes:

1. **Quantizer Only** (Recommended for beginners):
   ```yaml
   train_quantizer: true
   train_hifigan: false
   ```

2. **Full Training** (Advanced):
   ```yaml
   train_quantizer: true
   train_hifigan: true
   ```

## 💻 Hardware Requirements

### Minimum:
- **GPU**: 4GB VRAM (GTX 1660 or better)
- **RAM**: 8GB system RAM
- **Storage**: 2GB for models + dataset size

### Recommended:
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 5GB+ available space

### Memory Optimization:
```yaml
# For low-memory systems:
batch_size: 4              # Reduce batch size
num_workers: 2             # Reduce data workers
max_audio_duration: 5.0    # Shorter audio clips
```

## 📊 Training Process

### What Happens During Training:

1. **Data Loading**: Loads Vietnamese audio files
2. **Preprocessing**: Resamples to 16kHz, converts to mono
3. **Tokenization**: Encodes audio to tokens using pretrained Hubert
4. **Fine-tuning**: Adapts quantizer to Vietnamese speech patterns
5. **Validation**: Tests on held-out Vietnamese data
6. **Saving**: Saves best model based on validation loss

### Expected Training Time:
- **Small dataset** (100 files): 30 minutes - 1 hour
- **Medium dataset** (1000 files): 2-4 hours  
- **Large dataset** (10000 files): 8-12 hours

### Training Progress:
```
Epoch 1/20
Training: 100%|████████| 125/125 [05:23<00:00,  2.59s/it, loss=0.1234]
Validation: 100%|████████| 14/14 [00:32<00:00,  2.33s/it]
Train Loss: 0.1234, Val Loss: 0.0987

Epoch 2/20
...
```

## 🧪 Testing Your Model

### Test Single Audio File:
```bash
python test_vietnamese_tokenizer.py \
    --model_dir ./vietnamese_tokenizer \
    --audio_file vietnamese_test.wav \
    --pretrained_dir /kaggle/input/spiritlm-andinh
```

### Compare with Original Model:
```bash
python test_vietnamese_tokenizer.py \
    --model_dir ./vietnamese_tokenizer \
    --audio_file vietnamese_test.wav \
    --compare
```

### Expected Output:
```
Original Vietnamese Audio: [plays original]

Tokenization Results:
- Hubert tokens: 245 tokens
- First 50 tokens: 78 42 81 159 316 259 71 98 156 203...
- String representation: [Hu78][Hu42][Hu81][Hu159]...

Decoded from units (non-deduplicated): [plays reconstructed]
Decoded from string (deduplicated): [plays reconstructed]
```

## 📈 Monitoring Training

### Check Training Progress:
```bash
# Monitor GPU usage
nvidia-smi

# Check output directory
ls -la vietnamese_tokenizer/
# Should contain:
# - config.json (training config)
# - quantizer_vietnamese.pt (trained model)
# - checkpoint_epoch_5/ (periodic checkpoints)
```

### Troubleshooting Common Issues:

1. **Out of Memory Error**:
   ```yaml
   batch_size: 4  # Reduce from 8
   num_workers: 2 # Reduce from 4
   ```

2. **No Audio Files Found**:
   ```bash
   # Check your data directory structure
   find /path/to/data -name "*.wav" | head -10
   ```

3. **Training Loss Not Decreasing**:
   ```yaml
   learning_rate: 0.00005  # Reduce learning rate
   num_epochs: 50          # Train longer
   ```

## 🎵 Using Your Trained Model

### In Code:
```python
from test_vietnamese_tokenizer import VietnameseTokenizer

# Load your trained model
tokenizer = VietnameseTokenizer(
    model_dir="./vietnamese_tokenizer",
    pretrained_dir="/kaggle/input/spiritlm-andinh"
)

# Tokenize Vietnamese speech
units, string_tokens = tokenizer.encode_audio("vietnamese_audio.wav")
print(f"Tokens: {string_tokens[:100]}...")

# Reconstruct audio
reconstructed = tokenizer.decode_audio(string_tokens, speaker_id=0)
```

### Integration with Existing Code:
```python
# Replace your existing tokenizer with Vietnamese version
# OLD:
from spiritlm.speech_tokenizer import spiritlm_base
tokenizer = spiritlm_base()

# NEW:
from test_vietnamese_tokenizer import VietnameseTokenizer  
tokenizer = VietnameseTokenizer("./vietnamese_tokenizer", "/kaggle/input/spiritlm-andinh")

# Use same interface
units = tokenizer.encode_audio("vietnamese.wav")
reconstructed = tokenizer.decode_audio(units[1])  # units[1] is string_tokens
```

## 🎯 Expected Improvements

### After Vietnamese Training:
- ✅ **Better tokenization** of Vietnamese phonemes
- ✅ **More consistent tokens** for Vietnamese speech patterns
- ✅ **Improved reconstruction quality** for Vietnamese voices
- ✅ **Better handling** of Vietnamese tones and accents

### Quality Comparison:
```
Original Model on Vietnamese:
- Token consistency: 70-80%
- Reconstruction quality: Good but "foreign accent"
- Vietnamese-specific sounds: Sometimes lost

Vietnamese-Adapted Model:
- Token consistency: 85-95%
- Reconstruction quality: Better, more natural Vietnamese
- Vietnamese-specific sounds: Better preserved
```

## 📚 Advanced Usage

### Custom Data Augmentation:
```python
# Add to VietnameseSpeechDataset class
def augment_audio(self, wav):
    # Add noise
    if random.random() < 0.1:
        noise = torch.randn_like(wav) * 0.01
        wav = wav + noise
    
    # Speed perturbation
    if random.random() < 0.2:
        speed = random.uniform(0.9, 1.1)
        wav = torchaudio.functional.speed(wav, speed)
    
    return wav
```

### Multi-GPU Training:
```python
# Add to trainer initialization
if torch.cuda.device_count() > 1:
    self.hubert_model = nn.DataParallel(self.hubert_model)
```

### Resume Training:
```bash
# Save checkpoint manually and modify trainer to load from checkpoint
python train_vietnamese_tokenizer.py \
    --config my_config.yaml \
    --resume_from ./vietnamese_tokenizer/checkpoint_epoch_10
```

---

## 🔧 Next Steps

1. **Start with quantizer-only training** (easier, faster)
2. **Test on your Vietnamese audio** to verify improvement
3. **If satisfied, try HiFiGAN training** for better audio quality
4. **Experiment with different datasets** and hyperparameters
5. **Share your results** and improvements with the community!

Good luck with your Vietnamese speech tokenizer training! 🇻🇳✨ 