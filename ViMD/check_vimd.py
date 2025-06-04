import dask.dataframe as dd

splits = {
    'train': 'data/train-*-of-*.parquet',
    'test': 'data/test-*-of-*.parquet',
    'valid': 'data/valid-*-of-*.parquet'
}
df_train= dd.read_parquet("hf://datasets/nguyendv02/ViMD_Dataset/" + splits["train"])
df_test= dd.read_parquet("hf://datasets/nguyendv02/ViMD_Dataset/" + splits["test"])
df_valid= dd.read_parquet("hf://datasets/nguyendv02/ViMD_Dataset/" + splits["valid"])
first_rows = df_train.head(1)
len(audio_dict['bytes']) # =3447168
import dask.dataframe as dd
import soundfile as sf
import IPython.display as ipd
import numpy as np
import io
import librosa

# (Re)load your Dask DataFrame
# df = dd.read_parquet("your_file.parquet")

# Wrap the raw bytes as a file‐like object
audio_bytes = audio_dict['bytes']
with io.BytesIO(audio_bytes) as f:
    # librosa will detect the WAV format and return (samples, sr)
    audio_array, sampling_rate = librosa.load(f, sr=None, mono=False)
sampling_rate #=44100

len(audio_array[1]) # 861781
