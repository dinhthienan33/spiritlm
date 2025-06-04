# SpiRit-LM: Interleaved Spoken and Written Language Model - Complete Workflow

## Overview

SpiRit-LM (Spirit Language Model) is a multimodal language model developed by Meta that can seamlessly process and generate both text and speech in an interleaved manner. The model preserves the expressivity of input prompts and can learn new tasks across modalities in a few-shot manner.

## Architecture Components

### 1. Core Model Architecture

**Base Foundation**: The model is built on **LLaMA-2 7B** as its foundation language model.

```python
# From spiritlm/model/spiritlm_model.py:163-170
self.model = LlamaForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16
).to(self.device)
self.tokenizer = LlamaTokenizer.from_pretrained(
    pretrained_model_name_or_path=path,
    add_bos_token=True,
    add_eos_token=False,
)
```

**Two Model Variants**:
- **Spirit LM Base (7B)**: Uses phonetic tokens only (HuBERT)
- **Spirit LM Expressive (7B)**: Adds pitch and style tokens for enhanced expressivity

### 2. Speech Tokenization Pipeline

The speech tokenization system converts raw audio to discrete tokens through multiple specialized encoders:

#### A. Phonetic Tokenizer (HuBERT)
- **Model**: HuBERT + Linear Quantizer (96M parameters)
- **Input**: Raw waveform
- **Output**: Phonetic tokens (vocabulary: 501 tokens)
- **Implementation**: `spiritlm/speech_tokenizer/hubert/`

```python
# From spiritlm/speech_tokenizer/hubert/hubert_tokenizer.py
class HubertTokenizer(nn.Module):
    def forward(self, x, separate_channels=False, dense=False):
        feats = self.get_dense_features(x)  # Extract features
        tokens = self.quantizer_model(feats)  # Quantize to discrete tokens
        return tokens
```

#### B. Pitch Tokenizer (VQ-VAE) - Expressive Only
- **Model**: VQ-VAE (0.2M parameters)
- **Input**: Extracted F0 (fundamental frequency)
- **Output**: Pitch tokens (64 tokens)
- **Implementation**: `spiritlm/speech_tokenizer/f0/`

#### C. Style Tokenizer (Wav2Vec2) - Expressive Only
- **Model**: Wav2Vec2 + Linear Projection (95M parameters)
- **Input**: Raw waveform
- **Output**: Style tokens (100 tokens)
- **Implementation**: `spiritlm/speech_tokenizer/style_encoder/`

#### D. Speech Decoder (HiFi-GAN)
- **Base Model**: 14M parameters (phonetic tokens → waveform)
- **Expressive Model**: 15M parameters (phonetic + pitch + style → waveform)
- **Implementation**: `spiritlm/speech_tokenizer/hifigan/`

### 3. Text Processing

**Text Tokenization**: Uses standard LLaMA tokenizer for text processing
- **Vocabulary**: 32,000 base tokens + speech tokens
- **Special Tokens**: `[Text]` and `[Speech]` for modality switching
- **Total Vocabulary Size**: 
  - Base: 32,512 tokens
  - Expressive: 32,768 tokens

### 4. Unified Token Space

The model operates in a unified token space combining:
- **Text tokens**: Standard LLaMA vocabulary (0-31,999)
- **Special tokens**: `[Text]` (32000), `[Speech]` (32001)
- **Phonetic tokens**: `[Hu0]` to `[Hu500]` (32002-32502)
- **Pitch tokens**: `[Pi0]` to `[Pi63]` (32503-32566) - Expressive only
- **Style tokens**: `[St0]` to `[St99]` (32567-32666) - Expressive only

## Workflow Components

### 1. Model Loading (`spiritlm/model/spiritlm_model.py`)

```python
# Load Spirit LM model
spirit_lm = Spiritlm("spirit-lm-base-7b")  # or "spirit-lm-expressive-7b"

# Components loaded:
# 1. LLaMA-2 7B language model
# 2. LLaMA tokenizer with extended vocabulary
# 3. Speech tokenizers (HuBERT, optionally F0 + Style)
# 4. Speech decoder (HiFi-GAN)
```

### 2. Input Processing

**Multimodal Input Handling**:
```python
# Input can be text, speech, or mixed
interleaved_inputs = [
    GenerationInput(content="Hello", content_type=ContentType.TEXT),
    GenerationInput(content="audio.wav", content_type=ContentType.SPEECH)
]
```

**Prompt Building**:
```python
# From spiritlm_model.py:182-200
def _build_prompt(self, generation_inputs, output_modality):
    prompts = []
    for gen_input in generation_inputs:
        if gen_input.content_type == ContentType.SPEECH:
            prompts.append("[Speech]")
            prompts.append(self.speech_tokenizer(gen_input.content))  # Convert to tokens
        elif gen_input.content_type == ContentType.TEXT:
            prompts.append("[Text]")
            prompts.append(gen_input.content)
    return "".join(prompts)
```

### 3. Generation Process

**Generation Modes**:
- `OutputModality.TEXT`: Generate text only
- `OutputModality.SPEECH`: Generate speech only  
- `OutputModality.ARBITRARY`: Generate mixed text/speech

**Token Filtering**:
```python
# From spiritlm/model/utils.py:97-126
def get_forbidden_tokens(generate_only_speech=False, generate_only_text=False):
    if generate_only_speech:
        forbidden_tokens += list(range(32000))  # Ban text tokens
    elif generate_only_text:
        forbidden_tokens += list(range(32002, 32503))  # Ban speech tokens
    return forbidden_tokens
```

### 4. Output Decoding

**Mixed Output Parsing**:
```python
# From spiritlm_model.py:246-346
def _parse_speech_and_text(self, generated_content):
    # Parse mixed text/speech tokens
    # Identify [Text] and [Speech] sections
    # Group consecutive tokens by modality
    return splits  # List of (content, modality_type) tuples
```

**Speech Synthesis**:
```python
# Convert speech tokens back to audio
resyn_wav = self.speech_tokenizer.decode(speech_tokens, speaker_id=2)
```

## Training Data (From MODEL_CARD.md)

The model was trained on a massive multimodal dataset:

| Dataset Type | Hours | Speech Tokens | Text Tokens |
|--------------|-------|---------------|-------------|
| Speech-only  | 458K  | 28.2B         | -           |
| Speech+Text  | 111K  | 7.0B          | 1.4B        |
| Text-only    | -     | -             | 307B        |

**Training Infrastructure**:
- **Duration**: October-December 2023
- **Compute**: 21K GPU hours per model on A100-80GB
- **Framework**: Custom training libraries on Meta's Research Clusters

## Evaluation Framework

### Speech-Text Sentiment Preservation (STSP) Benchmark

**Purpose**: Evaluate sentiment preservation across modalities
**Task**: Generate text/speech that preserves input sentiment

**Evaluation Pipeline**:

1. **Data Preparation** (`spiritlm/eval/load_data.py`):
```python
# Load evaluation datasets
eval_dataset = SpeechData(manifest_path, root_dir=STSP_DATA_ROOT)
# or
eval_dataset = TextData(manifest_path, root_dir=STSP_DATA_ROOT)
```

2. **Prediction** (`spiritlm/eval/stsp/predict_stsp.py`):
```bash
# Speech to Text
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py \
  --model spirit-lm-base-7b \
  --eval_manifest_path data/stsp_data/manifest/emov/emov.test.jsonl \
  --input_output speech_text \
  --write_pred ./pred_s_t.jsonl
```

3. **Automatic Evaluation** (`spiritlm/eval/eval_stsp.py`):
```python
# Use sentiment/emotion classifiers to evaluate output
def eval(gold_records, predictions):
    # Compare predicted sentiment with reference
    # Calculate accuracy score
    return accuracy_percentage
```

**Evaluation Datasets**:
- **EMOV**: Emotional speech dataset
- **EXPRESSO-ASR**: Conversational speech
- **EXPRESSO-READ**: Read speech

## Key Implementation Files

### Core Model
- `spiritlm/model/spiritlm_model.py`: Main model class and generation logic
- `spiritlm/model/utils.py`: Utility functions for token processing

### Speech Processing
- `spiritlm/speech_tokenizer/spiritlm_tokenizer.py`: Main speech tokenizer
- `spiritlm/speech_tokenizer/hubert/`: HuBERT phonetic tokenizer
- `spiritlm/speech_tokenizer/f0/`: Pitch tokenizer (VQ-VAE)
- `spiritlm/speech_tokenizer/style_encoder/`: Style tokenizer (Wav2Vec2)
- `spiritlm/speech_tokenizer/hifigan/`: Speech decoder

### Evaluation
- `spiritlm/eval/eval_stsp.py`: Main evaluation script
- `spiritlm/eval/stsp/predict_stsp.py`: Prediction generation
- `spiritlm/eval/stsp/sentiment_classifiers.py`: Automatic evaluation models

### Examples
- `examples/speech_generation/`: Generation examples
- `examples/speech_tokenizer/`: Tokenization examples

## Usage Examples

### Basic Generation
```python
from spiritlm.model.spiritlm_model import Spiritlm, OutputModality, GenerationInput, ContentType

# Load model
spirit_lm = Spiritlm("spirit-lm-base-7b")

# Generate text from text
output = spirit_lm.generate(
    output_modality=OutputModality.TEXT,
    interleaved_inputs=[GenerationInput(
        content="The weather is", 
        content_type=ContentType.TEXT
    )]
)

# Generate speech from speech
output = spirit_lm.generate(
    output_modality=OutputModality.SPEECH,
    interleaved_inputs=[GenerationInput(
        content="path/to/audio.wav", 
        content_type=ContentType.SPEECH
    )]
)
```

### Speech Tokenization
```python
from spiritlm.speech_tokenizer import spiritlm_base

# Initialize tokenizer
tokenizer = spiritlm_base()

# Encode audio to tokens
units = tokenizer.encode_units("audio.wav")
# {'audio': 'path', 'hubert': '99 49 38 149...'}

# Encode to string format
tokens = tokenizer.encode_string("audio.wav")
# '[Hu99][Hu49][Hu38][Hu149]...'

# Decode back to audio
wav = tokenizer.decode(tokens, speaker_id=2)
```

This workflow demonstrates how SpiRit-LM bridges the gap between text and speech processing through a unified token space, enabling seamless multimodal language modeling. 