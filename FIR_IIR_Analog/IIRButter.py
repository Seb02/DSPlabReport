import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqz


sample_rate = 48000  
nyquist = sample_rate / 2  
fc = 1000  
normalized_fc = fc / nyquist  
filter_orders = [1, 2, 3, 4, 5]  # Orders of the Butterworth filter
attenuation_frequency = 10 * fc 


colors = plt.cm.viridis(np.linspace(0, 1, len(filter_orders)))

plt.figure(figsize=(12, 8))

# Iterate over filter orders
for idx, order in enumerate(filter_orders):
    
    b, a = iirfilter(
        N=order,
        Wn=normalized_fc,
        btype='low',
        analog=False,
        ftype='butter',
        output='ba'
    )
    
  
    w, h = freqz(b, a, worN=8000)
    freqs = w * nyquist / np.pi  # Convert to Hz
    response_dB = 20 * np.log10(np.abs(h))
    
    
    color = colors[idx]
    
    
    plt.plot(freqs, response_dB, label=f'Order {order}', color=color)
    
  
    attenuation_idx = np.argmin(np.abs(freqs - attenuation_frequency))  # Closest index to 10 * fc
    attenuation = response_dB[attenuation_idx]
    
   
    passband_idx = freqs <= fc  # Passband region is from 0 to fc
    passband_response_dB = response_dB[passband_idx]
    passband_ripple = np.max(passband_response_dB) - np.min(passband_response_dB)
    
    
    plt.scatter(attenuation_frequency, attenuation, color=color, label=f'Attenuation @ {attenuation_frequency} Hz): {attenuation:.2f} dB')
    plt.plot([], [], label=f'Passband Ripple (Order {order}): {passband_ripple:.2f} dB')


plt.title('Frequency Response of Butterworth Low-Pass Filters')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.axvline(fc, color='red', linestyle='--', label=f'Cutoff Frequency: {fc} Hz')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
