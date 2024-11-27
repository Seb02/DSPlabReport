import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter


sample_rate, audio_data = wav.read('/Users/sebastien/Desktop/DSP/speech_three-tones.wav')

# 0 = left
if len(audio_data.shape) > 1:
    mono_audio = audio_data[:, 0]  
else:
    mono_audio = audio_data

# Convert audio to float 
mono_audio = mono_audio.astype(np.float32)

# Normalize the audio
mono_audio /= np.max(np.abs(mono_audio), axis=0)


nyquist = sample_rate / 2  # Nyquist freq

# passband freq
lowcut = 315   
highcut = 2500  

# Normalized cutoff frequencies 
low = lowcut / nyquist
high = highcut / nyquist

# Filter order 
numtaps = 601 
window_type = 'blackman'  # Options: 'hann', 'hamming','blackman', 'kaiser', 'barlett', 'rectangular'
beta = 8  # β value for the Kaiser window

#tester les autres fonctions de génération des FIR : les autres que firwin (remez et firls)

fir_coeff = firwin(numtaps, [low, high], pass_zero=False, window=window_type)

#fir_coeff = firwin(numtaps, [low, high], pass_zero=False, window=(window_type, beta))
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
plt.show()

print(f"Filtered audio saved to {output_path}")
