import numpy as np
import scipy.io.wavfile as wavfile
import librosa

# when using the dataset from torch, this has already been done.

# # Load the audio file
# sample_rate, audio_data = wavfile.read('audio_file.wav')

# # Apply the same preprocessing steps as the TensorFlow code
# audio_samples = audio_data.astype(np.float32)
# audio_samples /= 32768.0

# # Resample the audio data to the desired sample rate
# resampled_audio_data = librosa.resample(audio_samples, sample_rate, 16000)

# Apply a pre-emphasis filter to the audio data
pre_emphasis = 0.97
emphasized_audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

# Convert the audio data to a spectrogram
# in the original preprocessing it's set to 480 and hop_l to 160, but cgpt suggests to use 640 and 320
spectrogram = np.abs(librosa.stft(emphasized_audio_data, n_fft=480, hop_length=160))

# Apply the microfrontend preprocessing method
window_size_ms = 30.0
window_stride_ms = 20.0
feature_bin_count = 40
dct_coefficient_count = 10
lower_frequency_limit = 20
upper_frequency_limit = 4000
sample_rate = 16000

# Apply the window function to the spectrogram
window_length_samples = int(round(window_size_ms * sample_rate / 1000))
window_stride_samples = int(round(window_stride_ms * sample_rate / 1000))
window = np.hanning(window_length_samples)
num_frames = 1 + int(np.floor((spectrogram.shape[1] - window_length_samples) / window_stride_samples))
# set up the array
mfccs = np.zeros((num_frames, dct_coefficient_count))

for frame in range(num_frames):
    start = frame * window_stride_samples
    end = start + window_length_samples
    windowed_spec = spectrogram[:, start:end] * window
    mfcc = librosa.feature.mfcc(
        S=windowed_spec,
        sr=sample_rate,
        n_mfcc=dct_coefficient_count,
        n_fft=640,
        hop_length=320,
        fmin=lower_frequency_limit,
        fmax=upper_frequency_limit
    )
    # Mel-Frequency Cepstral Coefficients.
    mfccs[frame, :] = mfcc

# Quantize the MFCC data
quantized_mfccs = (np.clip(mfccs, -80.0, 16.0) + 80.0) * (255.0 / 96.0)
quantized_mfccs = np.round(quantized_mfccs).astype(np.uint8)

# Reshape the MFCC data to the required shape (num_frames, num_dct_coeffs)
num_dct_coeffs = quantized_mfccs.shape[1]
reshaped_mfccs = quantized_mfccs.reshape((1, num_frames, num_dct_coeffs))

# Apply the window stride
window_stride = 20
start = 0
end = window_stride * reshaped_mfccs.shape[1]
stride_mfccs = reshaped_mfccs[:, start:end:window_stride, :]

# Convert the MFCC data to a tensor
mfcc_tensor = np.array(stride_mfccs, dtype=np.float32)

# Normalize the data to have zero mean and unit variance
mean = np.mean(mfcc_tensor)
std = np.std(mfcc_tensor)
normalized_mfcc_tensor = (mfcc_tensor - mean) / std