import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import remez, freqz, lfilter


sample_rate, audio_data = wav.read('/Users/sebastien/Desktop/DSP/speech_three-tones.wav')

if len(audio_data.shape) > 1:
    mono_audio = audio_data[:, 0]  
else:
    mono_audio = audio_data

# Convert audio to float and normalize
mono_audio = mono_audio.astype(np.float32)
mono_audio /= np.max(np.abs(mono_audio), axis=0)

nyquist = sample_rate / 2  # Nyquist frequency
fs = 2 * nyquist           # Set fs as the sampling frequency


band = [2000, 5000]    
trans_width = 260       
numtaps = 201           

# Define band edges for band-pass filter
edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, nyquist]

# Desired amplitude in each band: 0 = stopband, 1 = passband
desired = [0, 1, 0]

# Generate filter coefficients
fir_coeff = remez(numtaps, edges, desired, fs=fs)


filtered_audio = lfilter(fir_coeff, 1.0, mono_audio)

# Normalize the filtered audio back to int16 format
filtered_audio = np.int16(filtered_audio / np.max(np.abs(filtered_audio)) * 32767)


output_path = '/Users/sebastien/Desktop/DSP/filtered_speech_three-tones_remez_bandpass.wav'
wav.write(output_path, sample_rate, filtered_audio)


w, h = freqz(fir_coeff, worN=2000, fs=fs)
plt.figure(figsize=(10, 6))
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title('Remez FIR Band-pass Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()
plt.axvline(band[0], color='green')  
plt.axvline(band[1], color='red')  
plt.show()

print(f"Filtered audio saved to {output_path}")
