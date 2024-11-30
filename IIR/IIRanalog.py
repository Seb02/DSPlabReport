import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqresp
import time 

start_time = time.perf_counter()

lowcut = 315    
highcut = 2500  

# Convert frequencies to radians per second for analog filter design
lowcut_rad = 2 * np.pi * lowcut
highcut_rad = 2 * np.pi * highcut


filter_order = 14  # Order of the filter
filter_type = 'cheby2'  # 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
btype = 'band'  # Filter type: 'low', 'high', 'band', 'stop'


b, a = iirfilter(
    N=filter_order,               # Order of the filter
    Wn=[lowcut_rad, highcut_rad], # Cutoff frequencies in rad/s
    rp=2,                         # Passband ripple (Chebyshev, elliptic only)
    rs=95,                        # Stopband attenuation (Chebyshev, elliptic only)
    btype=btype,                  # Filter type
    analog=True,                  # Analog filter
    ftype=filter_type,            # Filter design type
    output='ba'                   # Output filter coefficients (numerator, denominator)
)

# Frequency range for the response (linear scale in Hz)
linear_freqs = np.linspace(0, 20000, 2000)  
angular_freqs = 2 * np.pi * linear_freqs  # Convert to rad/s

# Frequency response of the analog filter
w, h = freqresp((b, a), w=angular_freqs)
end_time = time.perf_counter()
# Plot the frequency response on a linear scale
plt.figure(figsize=(12, 6))  # Wider figure
plt.plot(linear_freqs, 20 * np.log10(abs(h)), 'b')  # Linear frequency scale
plt.title(f'{filter_type.capitalize()} Analog Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.axvline(lowcut, color='green', label=f'Lowcut: {lowcut} Hz')
plt.axvline(highcut, color='red', label=f'Highcut: {highcut} Hz')
plt.xlim([0, 20000])  # Set x-axis limits up to 20 kHz
plt.legend()
plt.show()

elapsed_time = end_time - start_time


print(f"Total computing time: {elapsed_time:.6f} seconds")
