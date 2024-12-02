import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Fixed parameters
sample_rate = 48000  
nyquist = sample_rate / 2 
fc = 1000  
normalized_fc = fc / nyquist  
numtaps_list = [55, 60, 65]  # Different filter lengths
beta = 8  # β value for the Kaiser window (only used for Kaiser)
attenuation_frequencies = [10 * fc]  # Frequencies to evaluate attenuation


window_color_map = {
    "hann": "blue",
    "hamming": "yellow",
    "blackman": "red",
    "kaiser": "green"
}

plt.figure(figsize=(12, 8))


for window, color in window_color_map.items():
    for numtaps in numtaps_list:
        # Generate FIR coefficients
        if window == 'kaiser':
            fir_coeff = firwin(numtaps, normalized_fc, pass_zero=True, window=(window, beta))
        else:
            fir_coeff = firwin(numtaps, normalized_fc, pass_zero=True, window=window)

        # Compute frequency response
        w, h = freqz(fir_coeff, worN=8000)
        freqs = w * nyquist / np.pi  # Convert to Hz
        response_dB = 20 * np.log10(np.abs(h))

        
        label = f'{window.capitalize()}, {numtaps} Taps'
        plt.plot(freqs, response_dB, label=label, color=color)

        # Mark attenuation at f = 10 * fc
        for freq in attenuation_frequencies:
            idx = np.argmin(np.abs(freqs - freq))  # Find closest index
            attenuation = response_dB[idx]


            attenuation_label = f'{window.capitalize()}, at {freq/1000} kHz: {attenuation:.2f} dB'
            plt.scatter(freq, attenuation, color=color)  
            plt.plot([], [], color=color, label=attenuation_label)  


plt.title('Frequency Response of FIR Low-Pass Filters')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.axvline(fc, color='black', linestyle='--', label=f'Cutoff: {fc} Hz')  
plt.legend()
plt.tight_layout()
plt.show()
