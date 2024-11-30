import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqz, lfilter


sample_rate, audio_data = wav.read('/Users/sebastien/Desktop/DSP/speech_three-tones.wav')

if len(audio_data.shape) > 1:
    mono_audio = audio_data[:, 0]  
else:
    mono_audio = audio_data


mono_audio = mono_audio.astype(np.float32)
mono_audio /= np.max(np.abs(mono_audio), axis=0)


nyquist = sample_rate / 2  # Nyquist frequency 
lowcut = 315   
highcut = 2500  

# Normalized frequencies for the IIR filter
low = lowcut / nyquist
high = highcut / nyquist


filter_order = 6 #4  # Order of the filter


filter_type = 'cheby2'  # 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
btype = 'band'  # Filter type: 'low', 'high', 'band', 'stop'


b, a = iirfilter(
    N=filter_order,         
    Wn=[low, high],          # Cutoff frequencies 
    rp=4,                    # Passband ripple : Chebyshev and elliptic filters    4
    rs=75,                   # Stopband attenuation : Chebyshev and elliptic filters   75
    btype=btype,             # Filter type ('low', 'high', 'band', 'stop')
    analog=False,            # Digital filter (not analog)
    ftype=filter_type,       # Filter design type 
    output='ba'              # Output filter coefficients forme numérateur dénominateur (forme sos : permet d'évincer la possibilité d'instabilité numérique)
    #step response pour vérifier l'instabilité
    #(sosfreqz)
)

# Apply the filter to the audio signal
filtered_audio = lfilter(b, a, mono_audio)


filtered_audio = np.int16(filtered_audio / np.max(np.abs(filtered_audio)) * 32767)

output_path = '/Users/sebastien/Desktop/DSP/filtered_speech_three-tones_iir.wav'
wav.write(output_path, sample_rate, filtered_audio)

# Frequency response of the filter 
w, h = freqz(b, a, worN=8000)


plt.figure(figsize=(10, 6))
plt.plot(w * nyquist / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title(f'{filter_type.capitalize()} IIR Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()
plt.axvline(lowcut, color='green', label=f'Lowcut: {lowcut} Hz')
plt.axvline(highcut, color='red', label=f'Highcut: {highcut} Hz')
plt.legend()
plt.show()

print(f"Filtered audio saved to {output_path}")


#mettre plusieurs filtres et changer la fréquence d'échantillonage 
