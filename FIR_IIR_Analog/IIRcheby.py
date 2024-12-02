import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqz


sample_rate = 48000  
nyquist = sample_rate / 2  
fc = 1000  
normalized_fc = fc / nyquist  
filter_orders = [1, 2, 3, 4, 5]  # Orders of the Chebyshev Type I filter
attenuation_frequency = 10 * fc 
ripple = 3 

plt.figure(figsize=(12, 8))


for order in filter_orders:
    
    b, a = iirfilter(
        N=order,
        Wn=normalized_fc,
        rp=ripple,  
        btype='low',
        analog=False,
        ftype='cheby1',
        output='ba'
    )
    
  
    w, h = freqz(b, a, worN=8000)
    freqs = w * nyquist / np.pi  
    response_dB = 20 * np.log10(np.abs(h))
    
    
    line, = plt.plot(freqs, response_dB, label=f'Order {order}')
    line_color = line.get_color()  
    
    
    idx = np.argmin(np.abs(freqs - attenuation_frequency))  
    attenuation = response_dB[idx]
    


    passband_idx = freqs <= fc
    passband_response_dB = response_dB[passband_idx]
    passband_ripple = np.max(passband_response_dB) - np.min(passband_response_dB)
    

    plt.scatter(attenuation_frequency, attenuation, color=line_color, 
                label=f'Attenuation @ {attenuation_frequency} Hz (Order {order}): {attenuation:.2f} dB')
    plt.plot([], [], label=f'Passband Ripple (Order {order}): {passband_ripple:.2f} dB')


plt.title('Frequency Response of Chebyshev Type I Low-Pass Filters')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.axvline(fc, color='red', linestyle='--', label=f'Cutoff Frequency: {fc} Hz')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
