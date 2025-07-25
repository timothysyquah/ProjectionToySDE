import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import lfilter, firwin

# --- Part 1: Core Filtering Logic (From Scratch using FFTs) ---

def fir_filter_with_fft(signal, kernel):
    """
    Applies an FIR filter to a signal using the FFT method (fast convolution).

    Args:
        signal (np.ndarray): The input signal to be filtered.
        kernel (np.ndarray): The impulse response of the FIR filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    # For linear convolution using FFTs, we need to pad both the signal and
    # the kernel to a combined length. The required length is N + M - 1,
    # where N is the length of the signal and M is the length of the kernel.
    signal_len = len(signal)
    kernel_len = len(kernel)
    final_len = signal_len + kernel_len - 1

    # 1. Compute the FFT of the signal, padded to the final length
    signal_fft = fft(signal, n=final_len)

    # 2. Compute the FFT of the kernel, padded to the final length
    kernel_fft = fft(kernel, n=final_len)

    # 3. Perform element-wise multiplication in the frequency domain
    #    This is equivalent to convolution in the time domain.
    filtered_fft = signal_fft * kernel_fft

    # 4. Compute the inverse FFT to get the filtered signal in the time domain
    filtered_signal = ifft(filtered_fft)

    # The result of convolution will have length `final_len`. We need to
    # trim it back to the original signal length.
    # The valid part of the linear convolution starts at index 0.
    return np.real(filtered_signal[:signal_len])


# --- Part 2: FIR Filter Kernel Generation (From Scratch) ---

def create_lowpass_kernel(cutoff_hz, fs, num_taps):
    """
    Creates a windowed-sinc low-pass FIR filter kernel.

    Args:
        cutoff_hz (float): The cutoff frequency in Hz.
        fs (int): The sampling frequency in Hz.
        num_taps (int): The number of taps (coefficients) in the filter.
                        Should be an odd number.

    Returns:
        np.ndarray: The low-pass filter kernel.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd number.")

    # Normalized cutoff frequency (from 0 to 0.5)
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist

    # Create the sinc function for an ideal low-pass filter
    t = np.arange(num_taps)
    center = (num_taps - 1) / 2
    
    # sinc(x) = sin(pi*x) / (pi*x)
    # We shift 't' so the center of the sinc is at the center of the kernel
    x = t - center
    sinc_kernel = np.sinc(x * normalized_cutoff)
    
    # Apply a window function (e.g., Blackman) to reduce ripples
    window = np.blackman(num_taps)
    windowed_sinc = sinc_kernel * window

    # Normalize the kernel so the gain at DC (0 Hz) is 1
    return windowed_sinc / np.sum(windowed_sinc)


def create_highpass_kernel(cutoff_hz, fs, num_taps):
    """
    Creates a high-pass FIR filter kernel using spectral inversion.

    Args:
        cutoff_hz (float): The cutoff frequency in Hz.
        fs (int): The sampling frequency in Hz.
        num_taps (int): The number of taps in the filter. Should be odd.

    Returns:
        np.ndarray: The high-pass filter kernel.
    """
    # 1. Create a low-pass kernel with the same parameters
    lp_kernel = create_lowpass_kernel(cutoff_hz, fs, num_taps)

    # 2. Use spectral inversion to get the high-pass kernel.
    #    This is done by subtracting the low-pass kernel from a delta function
    #    (an impulse at the center of the kernel).
    center = (num_taps - 1) / 2
    impulse = np.zeros(num_taps)
    impulse[int(center)] = 1.0
    
    hp_kernel = impulse - lp_kernel
    return hp_kernel


# --- Part 3: Demonstration and Validation ---

if __name__ == '__main__':
    # --- Setup ---
    fs = 1000  # Sampling frequency in Hz
    duration = 2  # Signal duration in seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Create a test signal with two frequencies: one low, one high
    low_freq = 20   # Hz
    high_freq = 200 # Hz
    signal = (np.sin(2 * np.pi * low_freq * t) +
              0.5 * np.sin(2 * np.pi * high_freq * t))

    # Filter design parameters
    num_taps = 101  # Must be odd for our kernel functions
    cutoff_hz = 80 # The frequency to separate low and high components

    # --- 1. Low-Pass Filtering Demo ---
    print("--- Testing Low-Pass Filter ---")

    # Create kernel with our from-scratch function
    lp_kernel_scratch = create_lowpass_kernel(cutoff_hz, fs, num_taps)

    # Filter the signal with our from-scratch FFT method
    lp_filtered_scratch = fir_filter_with_fft(signal, lp_kernel_scratch)

    # For validation: use SciPy's standard lfilter
    # Note: firwin generates a similar kernel to our scratch function
    lp_kernel_scipy = firwin(num_taps, cutoff_hz, fs=fs, pass_zero='lowpass')
    lp_filtered_scipy = lfilter(lp_kernel_scipy, 1.0, signal)

    # Plotting Low-Pass Results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Low-Pass Filtering Example (Cutoff = {} Hz)'.format(cutoff_hz), fontsize=16)

    axs[0].plot(t, signal, label='Original Signal')
    axs[0].set_title('Original Signal ({} Hz + {} Hz)'.format(low_freq, high_freq))
    axs[0].legend()

    axs[1].plot(t, lp_filtered_scratch, label='Filtered (Scratch FFT Method)', color='r')
    axs[1].set_title('Filtered with Scratch Implementation')
    axs[1].legend()

    axs[2].plot(t, lp_filtered_scipy, label='Filtered (SciPy lfilter)', color='g')
    axs[2].set_title('Filtered with SciPy for Validation')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- 2. High-Pass Filtering Demo ---
    print("\n--- Testing High-Pass Filter ---")

    # Create kernel with our from-scratch function
    hp_kernel_scratch = create_highpass_kernel(cutoff_hz, fs, num_taps)

    # Filter the signal with our from-scratch FFT method
    hp_filtered_scratch = fir_filter_with_fft(signal, hp_kernel_scratch)

    # For validation: use SciPy's standard lfilter
    hp_kernel_scipy = firwin(num_taps, cutoff_hz, fs=fs, pass_zero='highpass')
    hp_filtered_scipy = lfilter(hp_kernel_scipy, 1.0, signal)

    # Plotting High-Pass Results
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('High-Pass Filtering Example (Cutoff = {} Hz)'.format(cutoff_hz), fontsize=16)

    axs[0].plot(t, signal, label='Original Signal')
    axs[0].set_title('Original Signal ({} Hz + {} Hz)'.format(low_freq, high_freq))
    axs[0].legend()

    axs[1].plot(t, hp_filtered_scratch, label='Filtered (Scratch FFT Method)', color='r')
    axs[1].set_title('Filtered with Scratch Implementation')
    axs[1].legend()

    axs[2].plot(t, hp_filtered_scipy, label='Filtered (SciPy lfilter)', color='g')
    axs[2].set_title('Filtered with SciPy for Validation')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # --- 3. Plotting Filter Frequency Responses ---
    
    # Calculate frequency responses of the kernels
    N_FFT = 2048 # Use a larger FFT for a smoother frequency plot
    freq_axis = fftfreq(N_FFT, 1/fs)
    
    lp_kernel_fft = fft(lp_kernel_scratch, N_FFT)
    hp_kernel_fft = fft(hp_kernel_scratch, N_FFT)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis[:N_FFT//2], 20 * np.log10(np.abs(lp_kernel_fft[:N_FFT//2])), label='Low-Pass Kernel')
    plt.plot(freq_axis[:N_FFT//2], 20 * np.log10(np.abs(hp_kernel_fft[:N_FFT//2])), label='High-Pass Kernel', linestyle='--')
    plt.axvline(cutoff_hz, color='k', linestyle=':', label=f'Cutoff Freq: {cutoff_hz} Hz')
    plt.title('Frequency Response of Generated Filter Kernels')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim(-60, 5)
    plt.legend()
    plt.grid(True)
    plt.show()