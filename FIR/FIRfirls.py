import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import firls, freqz, lfilter


sample_rate, audio_data = wav.read('/Users/sebastien/Desktop/DSP/speech_three-tones.wav')


if len(audio_data.shape) > 1:
    mono_audio = audio_data[:, 0]  
else:
    mono_audio = audio_data

# Convert audio to float 
mono_audio = mono_audio.astype(np.float32)

# Normalize the audio
mono_audio /= np.max(np.abs(mono_audio), axis=0)

nyquist = sample_rate / 2  # Nyquist frequency

# Passband frequencies
lowcut = 315   
highcut = 2500  

# Normalized cutoff frequencies 
low = lowcut / nyquist
high = highcut / nyquist

# Filter order 
numtaps = 201


# Frequency bands and desired gains (0-1 scale)
bands = [0, low, low, high, high, 1]  # Frequency pairs for stopband, passband, and stopband
gains = [0, 0, 1, 1, 0, 0]       # Desired gains (0 in stopband, 1 in passband)

# Design the filter using firls
fir_coeff = firls(numtaps, bands, gains)

# Apply the filter 
filtered_audio = lfilter(fir_coeff, 1.0, mono_audio)

# Normalize the filtered audio back to int16 format 
filtered_audio = np.int16(filtered_audio / np.max(np.abs(filtered_audio)) * 32767)


output_path = '/Users/sebastien/Desktop/DSP/filtered_speech_three-tones.wav'
wav.write(output_path, sample_rate, filtered_audio)

# Frequency response of the filter
w, h = freqz(fir_coeff, worN=8000)

plt.figure(figsize=(10, 6))
plt.plot(w * nyquist / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()
plt.axvline(lowcut, color='green')  
plt.axvline(highcut, color='red')  
plt.xlim(0, sample_rate / 2)  # Limit x-axis to Nyquist frequency
plt.ylim(-100, 20)  # Set y-axis limits for better visualization
plt.show()

print(f"Filtered audio saved to {output_path}")
