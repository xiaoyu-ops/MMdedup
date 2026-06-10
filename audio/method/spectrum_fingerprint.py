import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.transform import resize
from tqdm import tqdm
import glob
import json

def create_binary_spectrogram(audio_file, output_file=None, show_plot=False):
    """
    Generate a binarized spectrogram fingerprint from an audio file.
    
    Args:
        audio_file: Input audio file path.
        output_file: Optional output image path.
    
    Returns:
        Binarized spectrogram array.
    """
    # Load the audio file.
    y, sr = librosa.load(audio_file, sr=None)
    
    # Compute the STFT.
    D = librosa.stft(y, n_fft=2048, hop_length=64)
    
    # Convert the complex STFT to a magnitude spectrum.
    magnitude = np.abs(D)
    
    # Use the log scale to emphasize weaker signals.
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Flip the frequency axis so low frequencies are at the bottom.
    log_magnitude = np.flipud(log_magnitude)
    
    # Apply thresholding and keep only high-energy regions.
    threshold = threshold_otsu(log_magnitude)
    binary_spectrogram = log_magnitude > (threshold + 6)  # Raise the threshold to reduce white speckles.
    
    # Resize to 128x32.
    binary_spectrogram_resized = resize(binary_spectrogram, (32, 128), 
                                         anti_aliasing=False, preserve_range=True).astype(bool)
    
    # Convert booleans to 0/1 values.
    binary_spectrogram_resized = binary_spectrogram_resized.astype(np.uint8)
    frequencyPeaks = np.reshape(binary_spectrogram_resized, (4096,))
    # Log in English to avoid encoding issues on Windows consoles
    print(f"Processed file {audio_file}: generated 4096-dim fingerprint vector")
    
    return frequencyPeaks

def process_wav_files(audio_dir, output_file="audio/binary_array_dict.npy"):
    """
    Process all WAV files under the given directory.
    
    Args:
        audio_dir: Directory containing WAV files.
        output_file: Output NumPy file path.
    """
    # Find all WAV files.
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in directory: {audio_dir}")
        return

    print(f"Found {len(wav_files)} WAV files in {audio_dir}")
    
    binary_array_dict = {}
    
    for index, wav_file in enumerate(tqdm(wav_files, desc="Processing WAV files")):
        try:
            # Process the WAV file directly without a temporary file.
            binary_spec = create_binary_spectrogram(wav_file, output_file=None, show_plot=False)
            
            # Use the filename as the key.
            filename = os.path.basename(wav_file)
            binary_array_dict[filename] = binary_spec
            # Alternative index key: binary_array_dict[index] = binary_spec
            
        except Exception as e:
            print(f"Failed to process file {wav_file}: {e}")
            continue
    
        print(f"Successfully processed {len(binary_array_dict)} files")
    
        # Save results
        np.save(output_file, binary_array_dict)
        print(f"Saved audio fingerprint vectors to: {output_file}")
    
    return binary_array_dict

def load_config_json(config_path):
    """Load a JSON config file and return None on errors."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Config file JSON error: {e}")
        return None
    
if __name__ == "__main__":

    config_path = r"D:\Deduplication_framework\audio\method\audio_config.json"
    data = load_config_json(config_path)
    # Directory containing WAV files.
    audio_directory = data.get("paths", {}).get("dataset_dir", "./audio/dataset")
    # audio_directory = "./audio/dataset"  # Replace with your WAV directory.
    
    # Process WAV files.
    result = process_wav_files(audio_directory)
    
    if result:
        print("Processing complete!")
        print(f"Generated fingerprint dictionary contains {len(result)} audio files")
