from datasets import load_dataset, load_from_disk
import joblib
import os
import numpy as np
import sounddevice as sd
import librosa

sampling_rate = 8000
def save_model(model, filename="random_forest_model.joblib"):
    """
    Saves the trained model to a file.

    Args:
        model: The trained scikit-learn model.
        filename (str): The name of the file to save the model to.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename="random_forest_model.joblib"):
    """
    Loads a trained model from a file.

    Args:
        filename (str): The name of the file to load the model from.

    Returns:
        The loaded scikit-learn model.
    """
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"Model file not found at {filename}")
        return None
    
def load_dataset_from_hf_and_save_local(save_path="./dataset"):
    if os.path.exists(save_path):
        print(f"Loading dataset from local path: {save_path}")
        dataset = load_from_disk(save_path)
    else:
        print(f"Dataset not found locally. Downloading and saving to {save_path}...")
        dataset = load_dataset("mteb/free-spoken-digit-dataset")
        dataset.save_to_disk(save_path)
        print(f"Dataset downloaded and saved to {save_path}.")
    return dataset

def add_noise(track, noise_factor=0.01):
    noise = np.random.randn(len(track)) * noise_factor
    return track + noise

# For simpler model, we can easily just compute the mean, but
# we are losing the temporal information
def apply_mfcc_to_dataset(dataset_split, sr, noise_factor, n_mfcc=13):
    """
    Applies MFCC (Mel-frequency cepstral coefficients) extraction to a dataset split.

    Args:
        dataset_split (Dataset): A split of the Hugging Face dataset (e.g., dataset["train"]).
        n_mfcc (int): The number of MFCCs to return.

    Returns:
        list: A list of MFCC features for each audio sample in the dataset split.
    """

    X = []
    y = []

    for track in dataset_split:
        audio_array = track["audio"]["array"]
        audio_array = add_noise(audio_array, noise_factor)
        label = track["label"]
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc, n_fft = 512, hop_length=128)
        mfccs_mean = np.mean(mfccs, axis=1)
        X.append(mfccs_mean)
        y.append(label)
    return np.array(X), np.array(y)

def record_audio(duration, fs, channels=1):
    """
    Records audio from the microphone and saves it to a WAV file.

    Args:
        filename (str): The name of the file to save the audio to.
        duration (int): The duration of the recording in seconds.
        fs (int): The sampling rate (e.g., 44100 for CD quality).
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
    """
    print(f"Recording for {duration} seconds with sampling rate {fs} Hz...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    audio_data = audio_data.flatten()
    return audio_data

def inference(audio_data, model):
    audio_data_float = audio_data.astype(float)
    recorded_mfccs = librosa.feature.mfcc(y=audio_data_float, sr=sampling_rate, n_mfcc=13, n_fft=512, hop_length=128)
    recorded_mfccs_flat = np.mean(recorded_mfccs, axis=1).reshape(1, -1)
    predicted_label = model.predict(recorded_mfccs_flat)
    print(f"\nPredicted digit for the recorded audio: {predicted_label[0]}")