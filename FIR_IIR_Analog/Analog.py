import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqresp
import time 

start_time = time.perf_counter()


fc = 1000  


fc_rad = 2 * np.pi * fc

filter_order = 4  
filter_type = 'cheby1'  # 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
btype = 'low'  # Filter type

b, a = iirfilter(
    N=filter_order,         
    Wn=fc_rad,               
    rp=3,                    # Passband ripple (Chebyshev, elliptic only)
    rs=65,                   # Stopband attenuation (Chebyshev, elliptic only)
    btype=btype,             # Filter type
    analog=True,             # Analog filter
    ftype=filter_type,       # Filter design type
    output='ba'              # Output filter coefficients (numerator, denominator)
)


linear_freqs = np.linspace(0, 20000, 2000)  
angular_freqs = 2 * np.pi * linear_freqs  # Convert to rad/s

# Frequency response of the analog filter
w, h = freqresp((b, a), w=angular_freqs)
end_time = time.perf_counter()


plt.figure(figsize=(12, 6))  # Wider figure
plt.plot(linear_freqs, 20 * np.log10(abs(h)), 'b')  
plt.title(f'{filter_type.capitalize()} Analog Low-Pass Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.axvline(fc, color='green', linestyle='--', label=f'Cutoff Frequency: {fc} Hz')
plt.xlim([0, 20000])  
plt.legend()
plt.show()

elapsed_time = end_time - start_time

print(f"Total computing time: {elapsed_time:.6f} seconds")
